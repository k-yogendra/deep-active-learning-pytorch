import numpy as np
import rasterio
from scipy.stats import entropy, skew, kurtosis
from skimage.feature import local_binary_pattern
import cv2

# Define feature extraction functions
def get_mean(band):
    return np.mean(band)

def get_var(band):
    return np.var(band)

def get_entropy(band):
    return entropy(band.flatten(), base=2)

def get_min(band):
    return np.min(band)

def get_max(band):
    return np.max(band)

def get_median(band):
    return np.median(band)

def get_range(band):
    return np.max(band) - np.min(band)

def get_percentile_25(band):
    return np.percentile(band, 25)

def get_percentile_75(band):
    return np.percentile(band, 75)

def coeff_var(band):
    return np.std(band) / np.mean(band)

def get_skew(band):
    return skew(band.flatten())

def get_kurt(band):
    return kurtosis(band.flatten())

def get_corr(band1, band2):
    flatten_band1 = band1.flatten()
    flatten_band2 = band2.flatten()
    return np.corrcoef(flatten_band1, flatten_band2)[0, 1]

def compute_lbp(band):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(band, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    hist_entropy = entropy(hist, base=2)
    return hist_entropy

# New feature functions
def get_std(band):
    return np.std(band)

def get_iqr(band):
    return np.percentile(band, 75) - np.percentile(band, 25)

def get_mad(band):
    return np.mean(np.abs(band - np.mean(band)))

def get_energy(band):
    return np.sum(np.square(band))

def get_contrast(band):
    return band.max() - band.min()

def get_homogeneity(band):
    return np.sum(np.abs(np.diff(band)))

def get_sum_of_absolutes(band):
    return np.sum(np.abs(band))

def get_zero_crossing_rate(band):
    return ((band[:-1] * band[1:]) < 0).sum()

def get_peak_to_peak(band):
    return np.ptp(band)

def get_fractal_dimension(band):
    return -1.5 * np.log(np.std(band) / np.mean(band))

def get_gini_coefficient(band):
    sorted_band = np.sort(band.flatten())
    n = len(sorted_band)
    cumulative_band = np.cumsum(sorted_band)
    return (2.0 / n) * (sum((i + 1) * sorted_band[i] for i in range(n)) / cumulative_band[-1]) - (n + 1) / n

def extract_features(infosys_buildings):
    red_band, green_band, blue_band, nir_band, re_band, coastal_band, yellow_band, pan_band, all_features = [], [], [], [], [], [], [], [], []
    hue_band, saturation_band, value_band, all_features_hsv = [], [], [], []
    light_band, gtor_band, btoy_band, all_features_lab = [], [], [], []
    rgb, red_three_band, green_three_band, blue_three_band = [], [], [], []
    hsv, hue_three_band, saturation_three_band, value_three_band = [], [], [], []
    lab, light_three_band, gtor_three_band, btoy_three_band = [], [], [], []
    rgb_hsv_lab = []

    functions = [get_mean, get_var, get_entropy, get_min, get_max, get_median, get_range, get_percentile_25, get_percentile_75, coeff_var, get_skew, get_kurt, get_std, get_iqr, get_mad, get_energy, get_contrast, get_homogeneity, get_sum_of_absolutes, get_zero_crossing_rate, get_peak_to_peak, get_fractal_dimension, get_gini_coefficient]

    for tiff_file_path in infosys_buildings:
        red, green, blue, nir, re, coastal, yellow, pan = ([] for _ in range(8))
        
        with rasterio.open(tiff_file_path) as src:
            tiff_data = src.read()
        try: 
            if np.unique(tiff_data[0]) == [0]:
                continue
        except ValueError:
            pass
        
        bands = [red, green, blue, nir, re, coastal, yellow, pan]
        for i in range(8):
            for func in functions:
                bands[i].append(func(tiff_data[i]))
        
        red_copied = red.copy()
        green_copied = green.copy()
        blue_copied = blue.copy()
        red_copied.append(get_corr(tiff_data[0], tiff_data[1]))
        red_copied.append(get_corr(tiff_data[0], tiff_data[2]))
        green_copied.append(get_corr(tiff_data[1], tiff_data[0]))
        green_copied.append(get_corr(tiff_data[1], tiff_data[2]))
        blue_copied.append(get_corr(tiff_data[2], tiff_data[0]))
        blue_copied.append(get_corr(tiff_data[2], tiff_data[1]))
        red_copied.append(compute_lbp(tiff_data[0]))
        green_copied.append(compute_lbp(tiff_data[1]))
        blue_copied.append(compute_lbp(tiff_data[2]))
        red_copied.extend(green_copied)
        red_copied.extend(blue_copied)
        rgb.append(red_copied)
        red_three_band.append(red_copied)
        green_three_band.append(green_copied)
        blue_three_band.append(blue_copied)
        
        for band in range(7):
            try:
                red.append(get_corr(tiff_data[0], tiff_data[band+1]))
            except:
                pass
            try:
                green.append(get_corr(tiff_data[1], tiff_data[band+2]))
            except:
                pass
            try:
                blue.append(get_corr(tiff_data[2], tiff_data[band+3]))
            except:
                pass
            try:
                nir.append(get_corr(tiff_data[3], tiff_data[band+4]))
            except:
                pass
            try:
                re.append(get_corr(tiff_data[4], tiff_data[band+5]))
            except:
                pass
            try:
                coastal.append(get_corr(tiff_data[5], tiff_data[band+6]))
            except:
                pass
            try:
                yellow.append(get_corr(tiff_data[6], tiff_data[band+7]))
            except:
                pass
        
        red.append(compute_lbp(tiff_data[0]))
        green.append(compute_lbp(tiff_data[1]))
        blue.append(compute_lbp(tiff_data[2]))
        nir.append(compute_lbp(tiff_data[3]))
        re.append(compute_lbp(tiff_data[4]))
        coastal.append(compute_lbp(tiff_data[5]))
        yellow.append(compute_lbp(tiff_data[6]))
        pan.append(compute_lbp(tiff_data[7]))

        red_copy = red.copy()
        red_copy.extend(green)
        red_copy.extend(blue)
        red_copy.extend(nir)
        red_copy.extend(re)
        red_copy.extend(coastal)
        red_copy.extend(yellow)
        red_copy.extend(pan)
        all_features.append(red_copy)

        red_band.append(red)
        green_band.append(green)
        blue_band.append(blue)
        nir_band.append(nir)
        re_band.append(re)
        coastal_band.append(coastal)
        yellow_band.append(yellow)
        pan_band.append(pan)

        hue, saturation, value = ([] for _ in range(3))

        rgb_image = cv2.merge([tiff_data[0], tiff_data[1], tiff_data[2]])
        rgb_image = np.uint8(rgb_image)
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        bands_hsv = [hue, saturation, value]

        for i in range(3):
            for func in functions:
                bands_hsv[i].append(func(hsv_image[:,:,i]))
        
        hue_copied = hue.copy()
        saturation_copied = saturation.copy()
        value_copied = value.copy()
        hue_copied.append(get_corr(hsv_image[:,:,0], hsv_image[:,:,1]))
        hue_copied.append(get_corr(hsv_image[:,:,0], hsv_image[:,:,2]))
        saturation_copied.append(get_corr(hsv_image[:,:,1], hsv_image[:,:,0]))
        saturation_copied.append(get_corr(hsv_image[:,:,1], hsv_image[:,:,2]))
        value_copied.append(get_corr(hsv_image[:,:,2], hsv_image[:,:,0]))
        value_copied.append(get_corr(hsv_image[:,:,2], hsv_image[:,:,1]))
        hue_copied.append(compute_lbp(hsv_image[:,:,0]))
        saturation_copied.append(compute_lbp(hsv_image[:,:,1]))
        value_copied.append(compute_lbp(hsv_image[:,:,2]))
        hue_copied.extend(saturation_copied)
        hue_copied.extend(value_copied)
        hsv.append(hue_copied)
        hue_three_band.append(hue_copied)
        saturation_three_band.append(saturation_copied)
        value_three_band.append(value_copied)
        
        hsv_image = np.concatenate((hsv_image, tiff_data[3][:, :, np.newaxis]), axis=-1)
        hsv_image = np.concatenate((hsv_image, tiff_data[4][:, :, np.newaxis]), axis=-1)
        hsv_image = np.concatenate((hsv_image, tiff_data[5][:, :, np.newaxis]), axis=-1)
        hsv_image = np.concatenate((hsv_image, tiff_data[6][:, :, np.newaxis]), axis=-1)
        hsv_image = np.concatenate((hsv_image, tiff_data[7][:, :, np.newaxis]), axis=-1)
        
        for band in range(7):
            try:
                hue.append(get_corr(hsv_image[:,:,0], hsv_image[:,:,band+1]))
            except:
                pass
            try:
                saturation.append(get_corr(hsv_image[:,:,1], hsv_image[:,:,band+2]))
            except:
                pass
            try:
                value.append(get_corr(hsv_image[:,:,2], hsv_image[:,:,band+3]))
            except:
                pass
        
        hue.append(compute_lbp(hsv_image[:,:,0]))
        saturation.append(compute_lbp(hsv_image[:,:,1]))
        value.append(compute_lbp(hsv_image[:,:,2]))
        
        hue_copy = hue.copy()
        hue_copy.extend(saturation)
        hue_copy.extend(value)
        hue_copy.extend(nir)
        hue_copy.extend(re)
        hue_copy.extend(coastal)
        hue_copy.extend(yellow)
        hue_copy.extend(pan)
        all_features_hsv.append(hue_copy)
        
        hue_band.append(hue)
        saturation_band.append(saturation)
        value_band.append(value)

        light, gtor, btoy = ([] for _ in range(3))

        rgb_image = cv2.merge([tiff_data[0], tiff_data[1], tiff_data[2]])
        rgb_image = np.uint8(rgb_image)
        lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)

        bands_lab = [light, gtor, btoy]

        for i in range(3):
            for func in functions:
                bands_lab[i].append(func(lab_image[:,:,i]))
        
        light_copied = light.copy()
        gtor_copied = gtor.copy()
        btoy_copied = btoy.copy()
        light_copied.append(get_corr(lab_image[:,:,0], lab_image[:,:,1]))
        light_copied.append(get_corr(lab_image[:,:,0], lab_image[:,:,2]))
        gtor_copied.append(get_corr(lab_image[:,:,1], lab_image[:,:,0]))
        gtor_copied.append(get_corr(lab_image[:,:,1], lab_image[:,:,2]))
        btoy_copied.append(get_corr(lab_image[:,:,2], lab_image[:,:,0]))
        btoy_copied.append(get_corr(lab_image[:,:,2], lab_image[:,:,1]))
        light_copied.append(compute_lbp(lab_image[:,:,0]))
        gtor_copied.append(compute_lbp(lab_image[:,:,1]))
        btoy_copied.append(compute_lbp(lab_image[:,:,2]))
        light_copied.extend(gtor_copied)
        light_copied.extend(btoy_copied)
        lab.append(light_copied)
        light_three_band.append(light_copied)
        gtor_three_band.append(gtor_copied)
        btoy_three_band.append(btoy_copied)

        lab_image = np.concatenate((lab_image, tiff_data[3][:, :, np.newaxis]), axis=-1)
        lab_image = np.concatenate((lab_image, tiff_data[4][:, :, np.newaxis]), axis=-1)
        lab_image = np.concatenate((lab_image, tiff_data[5][:, :, np.newaxis]), axis=-1)
        lab_image = np.concatenate((lab_image, tiff_data[6][:, :, np.newaxis]), axis=-1)
        lab_image = np.concatenate((lab_image, tiff_data[7][:, :, np.newaxis]), axis=-1)
        
        for band in range(7):
            try:
                light.append(get_corr(lab_image[:,:,0], lab_image[:,:,band+1]))
            except:
                pass
            try:
                gtor.append(get_corr(lab_image[:,:,1], lab_image[:,:,band+2]))
            except:
                pass
            try:
                btoy.append(get_corr(lab_image[:,:,2], lab_image[:,:,band+3]))
            except:
                pass
        
        light.append(compute_lbp(lab_image[:,:,0]))
        gtor.append(compute_lbp(lab_image[:,:,1]))
        btoy.append(compute_lbp(lab_image[:,:,2]))
        
        light_copy = light.copy()
        light_copy.extend(gtor)
        light_copy.extend(btoy)
        light_copy.extend(nir)
        light_copy.extend(re)
        light_copy.extend(coastal)
        light_copy.extend(yellow)
        light_copy.extend(pan)
        all_features_lab.append(light_copy)
                
        light_band.append(light)
        gtor_band.append(gtor)
        btoy_band.append(btoy)
        
        rgb_hsv_lab.append(red + green + blue + nir + re + coastal + yellow + pan + hue + saturation + value + light + gtor + btoy)
    
    return red_band, green_band, blue_band, nir_band, re_band, coastal_band, yellow_band, pan_band, all_features, hue_band, saturation_band, value_band, all_features_hsv, light_band, gtor_band, btoy_band, all_features_lab, rgb_hsv_lab, rgb, red_three_band, green_three_band, blue_three_band, hsv, hue_three_band, saturation_three_band, value_three_band, lab, light_three_band, gtor_three_band, btoy_three_band

def get_features(infosys_buildings):
    red_cool, green_cool, blue_cool, nir_cool, re_cool, coastal_cool, yellow_cool, pan_cool, all_features_cool, hue_cool, saturation_cool, value_cool, all_features_hsv_cool, light_cool, gtor_cool, btoy_cool, all_features_lab_cool, rgb_hsv_lab_cool, rgb, red_three_band, green_three_band, blue_three_band, hsv, hue_three_band, saturation_three_band, value_three_band, lab, light_three_band, gtor_three_band, btoy_three_band = extract_features(infosys_buildings)

    all_features_cool = np.array(all_features_cool)
    red_cool = np.array(red_cool)
    green_cool = np.array(green_cool)
    blue_cool = np.array(blue_cool)
    nir_cool = np.array(nir_cool)
    re_cool = np.array(re_cool)
    coastal_cool = np.array(coastal_cool)
    yellow_cool = np.array(yellow_cool)
    pan_cool = np.array(pan_cool)
    
    hue_cool = np.array(hue_cool)
    saturation_cool = np.array(saturation_cool)
    value_cool = np.array(value_cool)
    all_features_hsv_cool = np.array(all_features_hsv_cool)
    
    light_cool = np.array(light_cool)
    gtor_cool = np.array(gtor_cool)
    btoy_cool = np.array(btoy_cool)
    all_features_lab_cool = np.array(all_features_lab_cool)
    
    rgb_hsv_lab_cool = np.array(rgb_hsv_lab_cool)
    
    rgb = np.array(rgb)
    hsv = np.array(hsv)
    lab = np.array(lab)
    
    return all_features_cool, red_cool, green_cool, blue_cool, nir_cool, re_cool, coastal_cool, yellow_cool, pan_cool, hue_cool, saturation_cool, value_cool, all_features_hsv_cool, light_cool, gtor_cool, btoy_cool, all_features_lab_cool, rgb_hsv_lab_cool, rgb, hsv, lab

