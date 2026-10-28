def select_features(df):

    columns_to_drop = [
        'EmployeeCount',
        'EmployeeNumber',
        'Over18',
        'StandardHours',
        'Attrition'


    ]

    X = df.drop(columns_to_drop, axis=1)

    if "Attrition" in df.columns:    ##user not able to see ...it will predict by model

        y = df["Attrition"].map({    
            "No": 0,
            "Yes": 1
        })

        return X, y

    return X