# ref: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html


from sklearn.preprocessing import StandardScaler
data = [[0,0],[0,0],[1,1],[1,1]]
scaler = StandardScaler()
scaler_data = scaler.fit(data) #compute the mean and std to be used for later scaling

scaler_data_attr = dir(scaler_data)
for attr, value in scaler_data.__dict__.items():
    print(attr , " - " , value)

""" 
    o/p:
    with_mean  -  True
    with_std  -  True
    copy  -  True
    n_samples_seen_  -  4
    mean_  -  [0.5 0.5]
    var_  -  [0.25 0.25]
    scale_  -  [0.5 0.5]

    @todo how to compute it mathematically?
    std = 
    mean = 
"""

print(scaler.transform(data)) #Perform standardization by centering and scaling

## encoding 

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc.fit(X)


# OneHotEncoder(categorical_features=None, categories=None,
#        dtype=<... 'numpy.float64'>, handle_unknown='ignore',
#        n_values=None, sparse=True)
    