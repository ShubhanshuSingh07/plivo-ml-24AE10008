import os
import pandas as pd
import numpy as np
import librosa
import scipy.stats as stats

FEATURE_COLUMNS = [
    "pause_index", "cum_duration", "trailing_voiced_ratio", "z_pitch_trailing", 
    "f0_slope_300_500", "z_energy_trailing", "rms_slope_300", "fade_out_ratio",
    "spec_cent_trend", "spec_roll_trend", "mfcc_delta_mean", "relative_speaking_rate",
    "global_speaking_rate", "zcr_trend_100", "plosive_burst_proxy"
]

def extract_features_for_pause(audio_path, pause_start, pause_index):
    """
    Extracts causal features strictly using audio from t=0 to pause_start.
    """
    y, sr = librosa.load(audio_path, sr=16000, duration=pause_start)
    
    if len(y) < 1600: 
        empty_feats = {k: np.nan for k in FEATURE_COLUMNS}
        empty_feats["pause_index"] = pause_index
        return empty_feats
        
    win_len = int(1.5 * sr)
    trailing_y = y[-win_len:]
    
    frame_len = 400 
    hop_len = 160
    n_frames = 1 + (len(trailing_y) - frame_len) // hop_len
    
    features = {}
    features["pause_index"] = pause_index
    
    global_y = y
    
    global_f0, global_voiced_flag, _ = librosa.pyin(global_y, fmin=80, fmax=400, 
                                                 sr=sr, frame_length=frame_len, hop_length=hop_len)
    
    global_voiced_f0 = global_f0[global_voiced_flag] if global_voiced_flag is not None else []
    global_f0_mean = np.mean(global_voiced_f0) if len(global_voiced_f0) > 0 else 0.0
    global_f0_std = np.std(global_voiced_f0) if len(global_voiced_f0) > 1 else 1.0
    
    global_rms = librosa.feature.rms(y=global_y, frame_length=frame_len, hop_length=hop_len)[0]
    global_rms_mean = np.mean(global_rms)
    global_rms_std = np.std(global_rms) if len(global_rms) > 1 else 1.0
    
    global_voiced_ratio = np.sum(global_voiced_flag) / len(global_voiced_flag) if global_voiced_flag is not None and len(global_voiced_flag) > 0 else 0.0
    features["cum_duration"] = pause_start


    f0, voiced_flag, _ = librosa.pyin(trailing_y, fmin=80, fmax=400, 
                                  sr=sr, frame_length=frame_len, hop_length=hop_len)
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else []
    
    features["trailing_voiced_ratio"] = np.sum(voiced_flag) / len(voiced_flag) if voiced_flag is not None and len(voiced_flag) > 0 else 0.0
    
    if len(voiced_f0) > 0:
        features["z_pitch_trailing"] = (np.mean(voiced_f0) - global_f0_mean) / global_f0_std
    else:
        features["z_pitch_trailing"] = np.nan
        
    ms_500_idx = int((0.5 * sr) / hop_len)
    ms_300_idx = int((0.3 * sr) / hop_len)
    f0_segment = f0[ms_300_idx:ms_500_idx]
    voiced_segment = voiced_flag[ms_300_idx:ms_500_idx] if voiced_flag is not None else None
    
    if voiced_segment is not None and len(voiced_segment) > 0:
        voiced_f0_seg = f0_segment[voiced_segment]
        if len(voiced_f0_seg) > 1:
            slope, _, _, _, _ = stats.linregress(np.arange(len(voiced_f0_seg)), voiced_f0_seg)
            features["f0_slope_300_500"] = slope
        else:
            features["f0_slope_300_500"] = np.nan
    else:
        features["f0_slope_300_500"] = np.nan

    rms = librosa.feature.rms(y=trailing_y, frame_length=frame_len, hop_length=hop_len)[0]
    features["z_energy_trailing"] = (np.mean(rms) - global_rms_mean) / global_rms_std
    
    ms_300_idx_rms = n_frames - int((0.3 * sr) / hop_len)
    if ms_300_idx_rms < 0: ms_300_idx_rms = 0
    rms_300 = rms[ms_300_idx_rms:]
    if len(rms_300) > 1:
        slope, _, _, _, _ = stats.linregress(np.arange(len(rms_300)), rms_300)
        features["rms_slope_300"] = slope
    else:
        features["rms_slope_300"] = np.nan
        
    ms_100_idx = n_frames - int((0.1 * sr) / hop_len)
    ms_200_idx = n_frames - int((0.2 * sr) / hop_len)
    if ms_200_idx < 0: ms_200_idx = 0
    if ms_100_idx > ms_200_idx:
        fade_out = np.mean(rms[ms_100_idx:]) / (np.mean(rms[ms_200_idx:ms_100_idx]) + 1e-8)
        features["fade_out_ratio"] = fade_out
    else:
        features["fade_out_ratio"] = np.nan

    spec_cent = librosa.feature.spectral_centroid(y=trailing_y, sr=sr, n_fft=frame_len, hop_length=hop_len)[0]
    spec_roll = librosa.feature.spectral_rolloff(y=trailing_y, sr=sr, n_fft=frame_len, hop_length=hop_len)[0]
    mfcc = librosa.feature.mfcc(y=trailing_y, sr=sr, n_mfcc=13, n_fft=frame_len, hop_length=hop_len)
    
    delta_width = min(9, mfcc.shape[1])
    if delta_width < 3:
        mfcc_delta = np.zeros_like(mfcc)
    else:
        if delta_width % 2 == 0: delta_width -= 1
        mfcc_delta = librosa.feature.delta(mfcc, width=delta_width)
    
    if len(spec_cent) > 1:
        slope_c, _, _, _, _ = stats.linregress(np.arange(len(spec_cent)), spec_cent)
        slope_r, _, _, _, _ = stats.linregress(np.arange(len(spec_roll)), spec_roll)
        features["spec_cent_trend"] = slope_c
        features["spec_roll_trend"] = slope_r
    else:
        features["spec_cent_trend"] = np.nan
        features["spec_roll_trend"] = np.nan
        
    features["mfcc_delta_mean"] = np.mean(np.abs(mfcc_delta[:5, :]))

    ms_500_idx_v = n_frames - int((0.5 * sr) / hop_len)
    if ms_500_idx_v < 0: ms_500_idx_v = 0
    local_voiced_ratio = np.sum(voiced_flag[ms_500_idx_v:]) / max(1, len(voiced_flag[ms_500_idx_v:])) if voiced_flag is not None else 0.0
    features["relative_speaking_rate"] = local_voiced_ratio / (global_voiced_ratio + 1e-8)
    features["global_speaking_rate"] = global_voiced_ratio

    zcr = librosa.feature.zero_crossing_rate(y=trailing_y, frame_length=frame_len, hop_length=hop_len)[0]
    
    zcr_100 = zcr[ms_100_idx:]
    if len(zcr_100) > 1:
        slope_zcr, _, _, _, _ = stats.linregress(np.arange(len(zcr_100)), zcr_100)
        features["zcr_trend_100"] = slope_zcr
    else:
        features["zcr_trend_100"] = np.nan
        
    zcr_200 = zcr[ms_200_idx:]
    zcr_baseline = np.mean(zcr) + 2 * np.std(zcr)
    features["plosive_burst_proxy"] = np.max(zcr_200) - zcr_baseline
    
    return {k: features.get(k, np.nan) for k in FEATURE_COLUMNS}

def build_feature_frame(data_dir, labels_df):
    """
    Iterates over the labels dataframe to extract features for every pause,
    returning the feature matrix X and metadata matrix M.
    """
    features_list = []
    
    for _, row in labels_df.iterrows():
        audio_path = os.path.join(data_dir, row["audio_file"])
        feats = extract_features_for_pause(audio_path, row["pause_start"], row["pause_index"])
        features_list.append(feats)

    X = pd.DataFrame(features_list)
    X = X[FEATURE_COLUMNS]
    
    M = labels_df.copy()
    
    return X, M