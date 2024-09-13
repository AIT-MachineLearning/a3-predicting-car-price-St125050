import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df.drop(columns=['torque'], inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def preprocess_data(df):
    def get_brand_name(car_name):
        return car_name.split(' ')[0].strip()

    def clean_data(value):
        if isinstance(value, str):
            value = value.split(' ')[0].strip()
        return float(value) if value else 0

    df['name'] = df['name'].apply(get_brand_name)
    df['mileage'] = df['mileage'].apply(clean_data)
    df['max_power'] = df['max_power'].apply(clean_data)
    df['engine'] = df['engine'].apply(clean_data)
    df['name'].replace([...], [...], inplace=True)
    df['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    df['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    df['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    df['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
