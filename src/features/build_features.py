class AirportFeatureEngineer:
    """
    Sklearn-style transformer for airport features.
    
    Features created:
    1. ORIGIN_AVG_DELAY - Target encoding (mean delay per airport)
    2. A_{AIRPORT} - One-hot encoding for top 10 airports
    3. IS_MAJOR - Binary indicator (major vs regional airport)
    4. ORIGIN_TRAFFIC - Traffic volume per airport
    
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
        # TODO: Inicijalizuj prazne atribute koji će čuvati naučene statistike
        self.airport_avg_delay_ = None
        self.global_mean_ = None
        self.top_10_airports_ = None 
        self.airport_traffic_ = None
        self.median_traffic_ = None
    
    def fit(self, train_data):
        """
        Learn statistics from training data.
        
        Args:
            train_data: Training dataframe with ARRIVAL_DELAY column
            
        Returns:
            self (for method chaining)
        """
        self.airport_avg_delay_ = train_data.groupby('ORIGIN_AIRPORT')['ARRIVAL_DELAY'].mean().round(2)
        self.global_mean_ = train_data['ARRIVAL_DELAY'].mean().round(2)
        self.top_10_airports_ = train_data['ORIGIN_AIRPORT'].value_counts().head(10).index.tolist()
        self.airport_traffic_ = train_data['ORIGIN_AIRPORT'].value_counts()
        self.median_traffic_ = self.airport_traffic_.median()

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
        if self.airport_avg_delay_ is None:
            raise ValueError("FeatureEngineer must be fitted before transform! Call .fit() first.")    
        data = data.copy()

        data['ORIGIN_AVG_DELAY'] = data['ORIGIN_AIRPORT'].map(self.airport_avg_delay_)
        data['ORIGIN_AVG_DELAY'] = data['ORIGIN_AVG_DELAY'].fillna(self.global_mean_)

        for airport in self.top_10_airports_:
            data[f'A_{airport}'] = (data['ORIGIN_AIRPORT'] == airport).astype(int)

        data['ORIGIN_TRAFFIC'] = data['ORIGIN_AIRPORT'].map(self.airport_traffic_)
        data['ORIGIN_TRAFFIC'] = data['ORIGIN_TRAFFIC'].fillna(self.median_traffic_)

        data['AIRPORT_TYPE'] = data['ORIGIN_AIRPORT'].apply(self._classify_airport)
        data['IS_MAJOR'] = (data['AIRPORT_TYPE'] == 'MAJOR').astype(int)

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
