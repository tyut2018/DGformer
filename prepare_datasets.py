import os
import argparse
import numpy as np

try:
    import pandas as pd
except ImportError:
    print("pandas is required: pip install pandas")
    exit(1)


DATASETS = {
    'ETTh1': ('ETTh1.csv', 'date', 'OT', 'Electricity Transformer Temperature - hourly, 7 vars'),
    'ETTh2': ('ETTh2.csv', 'date', 'OT', 'Electricity Transformer Temperature - hourly, 7 vars'),
    'ETTm1': ('ETTm1.csv', 'date', 'OT', 'Electricity Transformer Temperature - 15min, 7 vars'),
    'ETTm2': ('ETTm2.csv', 'date', 'OT', 'Electricity Transformer Temperature - 15min, 7 vars'),
    'electricity': ('electricity.csv', 'date', None, 'Electricity consumption, 321 clients'),
    'traffic': ('traffic.csv', 'date', None, 'Traffic occupancy rates, 862 sensors'),
    'weather': ('weather.csv', 'date', 'OT', 'Weather station, 21 vars'),
    'exchange_rate': ('exchange_rate.csv', 'date', None, 'Exchange rates, 8 currencies'),
    'solar_energy': ('solar_AL.csv', None, None, 'Solar power output, 137 stations'),
}

SPLITS = {
    'ETTh1': (0.6, 0.2),
    'ETTh2': (0.6, 0.2),
    'ETTm1': (0.6, 0.2),
    'ETTm2': (0.6, 0.2),
    'electricity': (0.7, 0.1),
    'traffic': (0.7, 0.1),
    'weather': (0.7, 0.1),
    'exchange_rate': (0.7, 0.1),
    'solar_energy': (0.7, 0.1),
}


def convert_csv_to_npy(csv_path, output_path, date_col='date'):
    df = pd.read_csv(csv_path)
    if date_col and date_col in df.columns:
        df = df.drop(columns=[date_col])
    df = df.select_dtypes(include=[np.number])
    data = df.values.astype(np.float32)
    T, N = data.shape
    data = data[:, :, np.newaxis]
    np.save(output_path, data)
    print(f"  Saved: {output_path} | shape={data.shape} (T={T}, N={N}, C=1)")
    
    return data.shape


def download_datasets(output_dir):
    import urllib.request
    ett_base = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
    
    urls = {
        'ETTh1.csv': f"{ett_base}/ETTh1.csv",
        'ETTh2.csv': f"{ett_base}/ETTh2.csv",
        'ETTm1.csv': f"{ett_base}/ETTm1.csv",
        'ETTm2.csv': f"{ett_base}/ETTm2.csv",
        'exchange_rate.csv': 'https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt',
        'solar_AL.csv': 'https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt',
    }
    
    csv_dir = os.path.join(output_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    
    for filename, url in urls.items():
        filepath = os.path.join(csv_dir, filename)
        if os.path.exists(filepath):
            print(f"  [Skip] {filename} already exists")
            continue
        print(f"  [Download] {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"  [OK] {filename}")
        except Exception as e:
            print(f"  [FAIL] {filename}: {e}")
    
    return csv_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./data')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--datasets', nargs='+', default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.download:
        print("\n=== Downloading datasets ===")
        csv_dir = download_datasets(args.output_dir)
    elif args.csv_dir:
        csv_dir = args.csv_dir
    else:
        for d in ['./data/csv', './dataset', '../dataset', './data']:
            if os.path.isdir(d):
                csv_dir = d
                break
        else:
            print("No CSV directory found. Use --csv_dir or --download")
            return
    
    print(f"\n=== CSV directory: {csv_dir} ===")
    print(f"=== Output directory: {args.output_dir} ===\n")
    
    datasets_to_process = args.datasets if args.datasets else DATASETS.keys()
    
    for name in datasets_to_process:
        if name not in DATASETS:
            print(f"[WARN] Unknown dataset: {name}")
            continue
        
        csv_file, date_col, target_col, desc = DATASETS[name]
        csv_path = os.path.join(csv_dir, csv_file)
        npy_path = os.path.join(args.output_dir, f'{name}.npy')
        
        if not os.path.exists(csv_path):
            alt_paths = [
                os.path.join(csv_dir, 'ETT-small', csv_file),
                os.path.join(csv_dir, name, csv_file),
            ]
            found = False
            for alt in alt_paths:
                if os.path.exists(alt):
                    csv_path = alt
                    found = True
                    break
            if not found:
                print(f"[SKIP] {name}: CSV not found at {csv_path}")
                continue
        
        print(f"[{name}] {desc}")
        shape = convert_csv_to_npy(csv_path, npy_path, date_col)
        
        train_r, val_r = SPLITS[name]
        T = shape[0]
        print(f"  Split: train={int(T*train_r)}, val={int(T*val_r)}, test={T - int(T*train_r) - int(T*val_r)}")
        print()
    
    print("=== Done! ===")
    print("\nDataset summary for run scripts:")
    print("  ETTh1/h2: train_ratio=0.6, val_ratio=0.2, N=7")
    print("  ETTm1/m2: train_ratio=0.6, val_ratio=0.2, N=7")
    print("  Exchange:  train_ratio=0.7, val_ratio=0.1, N=8")
    print("  Solar:     train_ratio=0.7, val_ratio=0.1, N=137")


if __name__ == '__main__':
    main()
