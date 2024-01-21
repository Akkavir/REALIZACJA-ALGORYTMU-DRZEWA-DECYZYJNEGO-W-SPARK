from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import os
import datetime
from pyspark.sql.functions import when, col

print(f"\n#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\n START \n#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#")

import sys

# Add the path where your library is located
sys.path.append('C:/Users/User/Desktop/INZ/dtreeviz')

import dtreeviz

def replace_tip_amount(df):
    df = df.withColumn("A4", when(col("A4") > 0, 1).otherwise(0))
    return df

def check_for_missing_values(df):
    cleaned_df = df.na.drop()

    missing_values = cleaned_df.select([spark_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])

    if missing_values.select("A1").head()["A1"] == 0:
        print("Brak brakujących wartości.")
    else:
        missing_values.show()

    return cleaned_df

def convert_categorical_to_numerical(df):
    indexers = [
        StringIndexer(inputCol=column, outputCol=column+"_index")
        for column in ["A1", "A2", "A3", "A4", "A5", "class"]
    ]

    for indexer in indexers:
        df = indexer.fit(df).transform(df)

    return df

def oversampling(indexed_df, seed):

    # Count the number of instances in each class
    class_counts = indexed_df.groupBy('class_index').count()

    # Find the minority and majority class labels
    minority_class = class_counts.orderBy(col('count').asc()).first()[0]
    majority_class = class_counts.orderBy(col('count').desc()).first()[0]

    # Oversample the minority class to match the count of the majority class
    minority_data = indexed_df.filter(indexed_df['class_index'] == minority_class)
    majority_count = class_counts.filter(class_counts['class_index'] == majority_class).first()[1]

    oversampled_minority = minority_data.sample(withReplacement=True, fraction=majority_count/minority_data.count(), seed=seed)

    # Concatenate the oversampled minority class with the original DataFrame
    balanced_df = indexed_df.unionAll(oversampled_minority)
    return balanced_df

def undersampling(indexed_df, seed):
    # Assuming 'indexed_df' is your DataFrame after categorical to numerical conversion
    # Undersample the majority class to match the count of the minority class
    class_counts = indexed_df.groupBy('class_index').count()

    # Find the minority and majority class labels
    minority_class = class_counts.orderBy(col('count').asc()).first()[0]
    majority_class = class_counts.orderBy(col('count').desc()).first()[0]

    # Get counts of instances in both minority and majority classes
    minority_count = class_counts.filter(col('class_index') == minority_class).first()[1]
    majority_count = class_counts.filter(col('class_index') == majority_class).first()[1]

    # Undersample the majority class to match the count of the minority class
    majority_data = indexed_df.filter(col('class_index') == majority_class)
    undersampled_majority = majority_data.sample(withReplacement=False, fraction=minority_count / majority_count, seed=seed)

    # Get the minority class data
    minority_data = indexed_df.filter(col('class_index') == minority_class)

    # Concatenate the undersampled majority class with the minority class
    balanced_df = minority_data.unionAll(undersampled_majority)
    return balanced_df


def display_top_n_models(n=10):
    print("Dostępne modele:")
    model_files = os.listdir("models")
    model_files = [file for file in model_files if file.startswith("model_")]
    model_files.sort(reverse=True)  # Sort models in reverse chronological order
    if not model_files:
        print("Folder models jest pusty")
    else:
        for i, model_file in enumerate(model_files[:n], 0):
            print(f"{i} | {model_file}")

def ask_user_for_choice():
    while True:
        try:
            choice = int(input("Wybierz numer modelu: "))
            return choice
        except ValueError:
            print("Model nie instnieje.")

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def train_decision_tree_model(train_data, max_depth=None, criterion=0):
        criterion_mapping = {0: 'gini', 1: 'entropy'}
        print(f"\n====================================\n Generowanie nowego modelu\n====================================\n")
        assembler = VectorAssembler(inputCols=["A1_index", "A2_index", "A3_index", "A4_index", "A5_index"],
                                        outputCol="assembled_features")

        dt = DecisionTreeClassifier(labelCol="class_index", featuresCol="assembled_features", maxDepth=max_depth, impurity=criterion_mapping[criterion])

        pipeline = Pipeline(stages=[assembler, dt])

        model = pipeline.fit(train_data)
        return model

def load_exisiting_model(choice):
        # Lista modeli w folderze "models"
        model_files = os.listdir("models")
        model_files = [file for file in model_files if file.startswith("model_")]
        model_files.sort(reverse=True)
        
        if 0 < choice <= len(model_files):
            selected_model = model_files[choice]

            loaded_model = DecisionTreeClassificationModel.load(f"models/{selected_model}")
            print(f"Załadowano: models/{loaded_model}")
            return loaded_model

def evaluate_model(model, test_data, seed, depth, criterion, sampling = "none"):

    evaluator = MulticlassClassificationEvaluator(
            labelCol="class_index", predictionCol="prediction", metricName="accuracy"
    )
    predictions = model.transform(test_data)
    accuracy = evaluator.evaluate(predictions)
    print(f"Dokładność: {accuracy}")
    criterion_mapping = {0: 'gini', 1: 'entropy'}

    # Printing the extracted details

    print(f"\tSeed podziału: {seed}")
    print(f"\tMax. głębokość drzewa: {depth}")
    print(f"\tKryterium: {criterion_mapping[criterion]}")
    print(f"\tSampling: {sampling}")

    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Construct the info to be saved in the desired format
    info_to_save = f"{current_date} | accuracy = {accuracy:.5f} | depth = {depth} | criterion = {criterion_mapping[criterion]} | seed = {seed} | sampling = {sampling}\n"

    # Save the info to a text file (append mode)
    with open("model_info.txt", "a") as file:
        file.write(info_to_save)
        print("Zapisano do model_info.txt")
    return accuracy


def tune_hyperparameters(train_data, test_data):
    dt = DecisionTreeClassifier(labelCol="class_index", featuresCol="features")
    param_grid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [5, 10, 15]) \
        .addGrid(dt.maxBins, [20, 30, 40]) \
        .build()

    evaluator = MulticlassClassificationEvaluator(
        labelCol="class_index", predictionCol="prediction", metricName="accuracy"
    )

    cross_val = CrossValidator(estimator=dt,
                               estimatorParamMaps=param_grid,
                               evaluator=evaluator,
                               numFolds=3)

    cv_model = cross_val.fit(train_data)

    best_model = cv_model.bestModel
    predictions = best_model.transform(test_data)
    best_accuracy = evaluator.evaluate(predictions)

    print(f"Najlepsza dokładność {best_accuracy}")
    return best_model, best_accuracy



def save_model(model, sampling = "none"):
    user_savechoice = input("Zapisać ten model? (y/n): ").lower()
    if user_savechoice == 'y' or user_savechoice == 't':
        # Get the current date and time 
        current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Save the trained model to a directory with the current date as the filename
        model_path = f"models/model_{sampling}_{current_date}"
        model.save(model_path)
        print(f"Zapisano jako: {model_path}")

from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

def visualize_tree(indexed_df, seed):
    # Konwersja
    pandas_df = indexed_df.toPandas()

    # Ekstrakcja cech i zmiennej docelowej
    features = pandas_df[['A1_index', 'A2_index', 'A3_index', 'A4_index', 'A5_index']]
    target = pandas_df['class']

    # Utworzenie listy klas
    class_names_list = list(map(str, target.unique()))
# Podział danych    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=seed)
# Trenowanie klasyfikatora
    model = DTC()
    model.fit(X_train, y_train)


# Wizualizacja
    collumn_names = ["passenger_count", "RatecodeID", "payment_type", "tip_amount", "airport_fee", "VendorID"]
    plt.figure(figsize=(12, 8))
    tree.plot_tree(model, feature_names=collumn_names, class_names=class_names_list, filled=True)
# Zapis
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    treesave_path = f"treespngs/decision_tree_{current_date}.png"
    plt.savefig(treesave_path, dpi=800)
    plt.show()



def main():
    # Inicjalizacja sesji Spark
    spark = SparkSession.builder \
        .appName("PySpark Parquet Data Classification") \
        .getOrCreate()

    # Wybór tylko niezbędnych kolumn
    columns_to_select = ["passenger_count", "RatecodeID", "payment_type", "tip_amount", "airport_fee", "VendorID"]

    # Wczytanie danych z pliku Parquet
    parquet_path = "yellow.parquet"
    df = spark.read.parquet(parquet_path).limit(100).select(columns_to_select)
    
    # Zmiana nazw kolumn
    new_column_names = ["A1", "A2", "A3", "A4", "A5", "class"]
    for old_name, new_name in zip(columns_to_select, new_column_names):
        df = df.withColumnRenamed(old_name, new_name)
    df = replace_tip_amount(df)


    from pyspark.sql.functions import col, countDistinct
    unique_counts = df.select([countDistinct(col(c)).alias(c) for c in df.columns])
    unique_counts.show()
    
    
    print(f"\n====================================\n Załadowana tablica:\n====================================\n")
    df.show()

    row_count = df.select(col("A1")).filter(col("A1").isNotNull()).count()
    print(f"Number of rows in: {row_count}")
    print(f"\n====================================\n Brakujące wartości:\n====================================\n")
    # Handle missing values
    df = check_for_missing_values(df)
    # Convert categorical to numerical
    indexed_df = convert_categorical_to_numerical(df)
    print(f"\n====================================\n Tablica z indexami:\n====================================\n")
    indexed_df.show()

    # Create the assembler and transform the indexed data
    assembler = VectorAssembler(inputCols=["A1_index", "A2_index", "A3_index", "A4_index", "A5_index"],
                                outputCol="features")
    indexed_df = assembler.transform(indexed_df)

    import random
    seed_input = input("Podaj seed (ENTER - losowy): ").lower()
    try:
        seed = int(seed_input)
    except ValueError:
        seed = random.randint(0, 9999)
        print(f"Seed: {seed}")
    train_data, test_data = indexed_df.randomSplit([0.7, 0.3] , seed=seed)

    choice = 0
    user_choice = input("Załadować model? (y/n): ").lower()
    if user_choice == 'y' or user_choice == 't':
        display_top_n_models(10)
        choice = ask_user_for_choice()
        # Train the model
        model = load_exisiting_model(choice)
        max_depth = None
        criterion = 0
    else:       
        max_depth_input = input("Podaj maksymalną głębokość drzewa: ").strip()

        # Check if max_depth_input is a valid number, otherwise set it to default
        try:
            max_depth = int(max_depth_input)
            if max_depth <= 0:
                max_depth = None
        except ValueError:
            max_depth = None

        criterion_input = input("Wybierz kryterium podziału wezłów: [0 - Gini][1 - Entropia]: ").strip()
        # Check if criterion_input is a valid number, otherwise set it to default
        try:
            criterion = int(criterion_input)
            if criterion not in [0, 1]:
                criterion = 0
        except ValueError:
            criterion = 0

        model = train_decision_tree_model(train_data, max_depth, criterion)     

    # Evaluate the model
    accuracy = evaluate_model(model, test_data, seed, max_depth, criterion)
    save_model(model)

    print(f"\n====================================\n Oversampling:")
    oversampled_df = oversampling(indexed_df, seed)
    if user_choice == 'y' or user_choice == 't':
        OVmodel = model
    else:
        OVmodel = train_decision_tree_model(train_data, max_depth, criterion)
    OVaccuracy = evaluate_model(OVmodel, test_data, seed, max_depth, criterion, "oversampling")
    save_model(OVmodel, "oversampling")

    print(f"\n====================================\n Undersampling:")
    undersampled_df = undersampling(indexed_df, seed)
    if user_choice == 'y' or user_choice == 't':
        USmodel = model
    else:       
        USmodel = train_decision_tree_model(train_data, max_depth, criterion)
    USaccuracy = evaluate_model(USmodel, train_data, seed, max_depth, criterion, "undersampling")
    save_model(USmodel, "undersampling")

    print(f"\n====================================\nHyperparameters:\n====================================\n")
    # Wywołanie funkcji
    hyper_model = tune_hyperparameters(train_data, test_data)

    visualize_tree(indexed_df, seed)

    save_model(hyper_model, "hiperparameters")
    spark.stop()
    SparkSession.stop(self=spark)

    
main()