import pandas as pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data
trainingData = pandas.read_csv("Data for ML/trainingData.csv")

trainingData = pandas.DataFrame(trainingData.values, columns=trainingData.columns)

# Define the inputs and targets
trainingData = trainingData.sample(n=2000000, random_state=42)
input = trainingData.drop(['IPV4_SRC_ADDR', 'IPV4_DST_ADDR','Label', 'Attack'], axis=1)
target = trainingData['Attack'].values.astype(float)

# Data split
input_training, inputs_test, target_training, target_test = train_test_split(input, target, test_size=0.2, random_state=42)

# Not using an Random Forest for project but it is being used as a test
rfcAttack = RandomForestClassifier(n_estimators=70, max_depth=80, bootstrap=True, min_samples_split=2, min_samples_leaf=1, random_state=42)
rfcAttack.fit(input_training, target_training)

AllPredictions= []
AllAttacks = []
MSE = []

for i in range(5):
    attackPrediction = rfcAttack.predict(inputs_test)
    print(attackPrediction[10])
    AllPredictions.append(attackPrediction)
    AllAttacks.append(target_test)
    
    errorMean = mean_squared_error(target_test, attackPrediction)
    
    print("MSE: ", errorMean)
    
    MSE.append(errorMean)
    
    input_training, inputs_test, target_training, target_test = train_test_split(input, target, test_size=0.2, random_state=42)

#Plot a graph
fig, ax = plt.subplots(figsize=(5, 5))

# Create box plot for MSE and set color
bp_coverage = ax.boxplot(MSE, patch_artist=True, boxprops=dict(facecolor='skyblue'))

# Set y-axis limit to customize the scale
ax.set_ylim(min(MSE) - 0.05, max(MSE) + 0.05)

# Set y-axis label and title
ax.set_ylabel('Mean Squared Error')
ax.set_title('Distribution of Mean Squared Errors for Coverage MSE')

# Set color for outliers
for flier in bp_coverage['fliers']:
    flier.set(marker='o', color='skyblue', alpha=0.5)  # Change color and transparency as needed

# Show the plot
plt.show()