class AirportFeatureEngineer:
    """
    Sklearn-style transformer for airport features.
    
    Features created:
    - ORIGIN_AVG_DELAY: Mean arrival delay for origin airport
    - DESTINATION_AVG_DELAY: Mean arrival delay for destination airport
    - OA_{AIRPORT}: One-hot encoding for top 10 origin airports
    - DA_{AIRPORT}: One-hot encoding for top 10 destination airports
    - AIRLINE_{CODE}: One-hot encoding for all airlines  # NEW
    - IS_MAJOR_ORIGIN: Binary indicator (major vs regional origin)
    - IS_MAJOR_DESTINATION: Binary indicator (major vs regional destination)
    - ORIGIN_TRAFFIC: Number of flights from origin airport
    - DESTINATION_TRAFFIC: Number of flights to destination airport
    
    Usage:
        # Training
        fe = AirportFeatureEngineer()
        fe.fit(train_data)
        
        train_transformed = fe.transform(train_data)
        val_transformed = fe.transform(val_data)
        
        # Production
        joblib.dump(fe, 'feature_engineer.pkl')
        fe_loaded = joblib.load('feature_engineer.pkl')
        new_data_transformed = fe_loaded.transform(new_data)
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        # Learned statistics (fitted on training data)
        self.origin_a_avg_delay_ = None
        self.destination_a_avg_delay_ = None
        self.global_mean_ = None
        self.origin_a_traffic_ = None
        self.destination_a_traffic_ = None
        self.airline_counts_ = None
    
    def fit(self, train_data):
        """
        Learn statistics from training data.
        
        Args:
            train_data: Training dataframe with ARRIVAL_DELAY column
            
        Returns:
            self (for method chaining)
        """
        self.origin_a_avg_delay_ = train_data.groupby('ORIGIN_AIRPORT')['ARRIVAL_DELAY'].mean().round(2)
        self.destination_a_avg_delay_ = train_data.groupby('DESTINATION_AIRPORT')['ARRIVAL_DELAY'].mean().round(2)
        self.global_mean_ = train_data['ARRIVAL_DELAY'].mean().round(2)

        self.origin_a_traffic_ = train_data['ORIGIN_AIRPORT'].value_counts()
        self.destination_a_traffic_ = train_data['DESTINATION_AIRPORT'].value_counts()

        self.airline_counts_ = train_data['AIRLINE'].value_counts()

        return self
    
    def _classify_airport(self, code):
        if code.isalpha() and len(code) == 3:
            return 'MAJOR'
        else:
            return 'REGIONAL'

    def transform(self, data):
        """
        Apply learned statistics to create features.
        
        Args:
            data: Dataframe to transform
            
        Returns:
            Transformed dataframe with new features
        """
        if self.origin_a_avg_delay_ is None:
            raise ValueError("FeatureEngineer must be fitted before transform! Call .fit() first.")    
        data = data.copy()

        # 1. Drop leakage columns (if they exist)
        leakage_cols = ['YEAR', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF',
                        'ELAPSED_TIME', 'AIR_TIME', 'WHEELS_ON', 'TAXI_IN', 'ARRIVAL_TIME', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY',
                        'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'CANCELLED', 'DIVERTED', 'CANCELLATION_REASON',
                        'SCHEDULED_TIME']
        data = data.drop(columns=[c for c in leakage_cols if c in data.columns], errors='ignore')
        
        # 2. Drop multicollinear
        if 'DISTANCE' in data.columns:
            data = data.drop('DISTANCE', axis=1)

        data['ORIGIN_AVG_DELAY'] = data['ORIGIN_AIRPORT'].map(self.origin_a_avg_delay_)
        data['ORIGIN_AVG_DELAY'] = data['ORIGIN_AVG_DELAY'].fillna(self.global_mean_)
        data['DESTINATION_AVG_DELAY'] = data['DESTINATION_AIRPORT'].map(self.destination_a_avg_delay_)
        data['DESTINATION_AVG_DELAY'] = data['DESTINATION_AVG_DELAY'].fillna(self.global_mean_)

        for airport in self.origin_a_traffic_.head(10).index.tolist():
            data[f'OA_{airport}'] = (data['ORIGIN_AIRPORT'] == airport).astype(int)
        
        for airport in self.destination_a_traffic_.head(10).index.tolist():
            data[f'DA_{airport}'] = (data['DESTINATION_AIRPORT'] == airport).astype(int)

        for airline in self.airline_counts_.index.tolist():
            data[f'AIRLINE_{airline}'] = (data['AIRLINE'] == airline).astype(int)

        data['ORIGIN_TRAFFIC'] = data['ORIGIN_AIRPORT'].map(self.origin_a_traffic_)
        data['ORIGIN_TRAFFIC'] = data['ORIGIN_TRAFFIC'].fillna(self.origin_a_traffic_.median())
        data['DESTINATION_TRAFFIC'] = data['DESTINATION_AIRPORT'].map(self.destination_a_traffic_)
        data['DESTINATION_TRAFFIC'] = data['DESTINATION_TRAFFIC'].fillna(self.destination_a_traffic_.median())

        data['ORIGIN_A_TYPE'] = data['ORIGIN_AIRPORT'].apply(self._classify_airport)
        data['DESTINATION_A_TYPE'] = data['DESTINATION_AIRPORT'].apply(self._classify_airport)
        data['IS_MAJOR_ORIGIN'] = (data['ORIGIN_A_TYPE'] == 'MAJOR').astype(int)
        data['IS_MAJOR_DESTINATION'] = (data['DESTINATION_A_TYPE'] == 'MAJOR').astype(int)

        data = data.drop(['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'ORIGIN_A_TYPE', 'DESTINATION_A_TYPE'], axis=1)

        return data
    
    def fit_transform(self, train_data):
        """
        Fit and transform in one step (shortcut for training data).
        
        Args:
            train_data: Training dataframe
            
        Returns:
            Transformed training dataframe
        """
        return self.fit(train_data).transform(train_data)
    

if __name__ == '__main__':
    import pandas as pd

    train = pd.read_csv("../../data/processed/train.csv")
    val = pd.read_csv("../../data/processed/val.csv")

    print("Testing AirportFeatureEngineer...")

    fe = AirportFeatureEngineer()
    fe.fit(train)
    train_transformed = fe.transform(train)
    val_transformed = fe.transform(val)

    print(f"\n✅ Train shape: {train_transformed.shape}")
    print(f"✅ Val shape: {val_transformed.shape}")
    print(f"\n✅ New features created: {[col for col in train_transformed.columns if col not in train.columns]}")
    
    # Test save/load
    import joblib
    joblib.dump(fe, "../../models/feature_engineer.pkl")
    print("\n✅ Feature engineer saved!")
    
    fe_loaded = joblib.load("../../models/feature_engineer.pkl")
    print("✅ Feature engineer loaded!")
