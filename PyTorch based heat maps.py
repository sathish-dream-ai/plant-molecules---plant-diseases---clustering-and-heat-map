import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 100)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(100, 50)
        self.output_layer = nn.Linear(50, 49)  # 49 clusters (0 to 48)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

# Train the model
def train_model(model, loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        logging.info(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Generate heatmap from detailed data CSV
def generate_heatmap(detailed_csv_path, molecule, disease, condition_path):
    detailed_data = pd.read_csv(detailed_csv_path, index_col=0)
    logging.debug(f"Detailed data read from CSV:\n{detailed_data.head()}")

    # Ensure that the data is in the correct format for heatmap
    count_data = detailed_data.select_dtypes(include='number')  # Select only numeric data
    logging.debug(f"Count data for heatmap:\n{count_data.head()}")

    # Generate and save the heatmap visualization
    plt.figure(figsize=(count_data.shape[1] * 2, count_data.shape[0] * 0.5))
    ax = sns.heatmap(count_data, annot=True, cmap='viridis_r', fmt='d', linewidths=0.5, linecolor='white', annot_kws={"size": 18})
    ax.invert_yaxis()

    # Set axis labels
    plt.xlabel('Name of the Plants', fontsize=14, labelpad=10)  # Add padding for the x-axis label
    plt.ylabel('Cluster Number', fontsize=14, labelpad=10)      # Add padding for the y-axis label

    # Title for the heatmap
    plt.title(f'Heatmap for {molecule}-{disease}', fontsize=16, pad=20)  # Add padding for the title

    plt.savefig(os.path.join(condition_path, f'heatmap_{molecule}_{disease}.png'), dpi=300)
    plt.close()
    logging.info(f"Heatmap visualization saved for {molecule}-{disease}.")

# Process data and generate heatmap
def process_data(base_directory, molecule, disease, first_csv, output_base_directory):
    """Process data to generate and save heatmaps, detailed CSV files, and merged output, ensuring all clusters are represented."""
    condition_path = f'{base_directory}/all_new/{molecule}/{molecule}-{disease}'
    output_condition_path = os.path.join(output_base_directory, f'{molecule}/{disease}')

    # Ensure the output directory exists
    os.makedirs(output_condition_path, exist_ok=True)
    logging.info(f"Directory ensured: {output_condition_path}")

    # Path to the second CSV file
    second_csv_path = os.path.join(condition_path, 'SMILES_Aligned_Groups_A_C.csv')
    if os.path.exists(second_csv_path):
        second_csv = pd.read_csv(second_csv_path)
        merge_columns = ['molecules', 'SMILES', 'plant', 'Dataset_Name']
        merged_data = pd.merge(second_csv, first_csv[merge_columns + ['Cluster']], on=merge_columns, how='inner')

        # Save the merged output CSV file
        merged_output_path = os.path.join(output_condition_path, f'merged_output_{molecule}_{disease}.csv')
        merged_data.to_csv(merged_output_path, index=False)
        logging.info(f"Merged data saved at {merged_output_path}.")

        # Assume maximum cluster number you expect is 48 (you can adjust this if necessary)
        all_clusters = pd.Index(range(49), name='Cluster')

        # Generate detailed data for heatmap (counting molecules and keeping names)
        detailed_data = (
            merged_data.groupby(['Cluster', 'plant'])['molecules']
            .agg(lambda x: ', '.join(x) if len(x) > 0 else '0')  # Join molecule names or use '0' if empty
        ).unstack(fill_value='0')

        # Ensure all clusters are included in the detailed data
        detailed_data = detailed_data.reindex(all_clusters, fill_value='0')

        # Create a count DataFrame for heatmap
        count_data = merged_data.groupby(['Cluster', 'plant'])['molecules'].count().unstack(fill_value=0)
        count_data = count_data.reindex(all_clusters, fill_value=0)  # Fill missing clusters with 0

        # Save detailed data CSV
        detailed_csv_path = os.path.join(output_condition_path, f'detailed_data_{molecule}_{disease}.csv')
        detailed_data.fillna(0, inplace=True)  # Fill NaN with 0 in detailed data
        detailed_data = detailed_data.astype(str)  # Ensure all data is string for CSV
        detailed_data.to_csv(detailed_csv_path)
        logging.info(f"Detailed data CSV saved for {molecule}-{disease}.")

        # Save the counts for heatmap
        heatmap_csv_path = os.path.join(output_condition_path, f'heatmap_data_{molecule}_{disease}.csv')
        count_data.to_csv(heatmap_csv_path)
        logging.info(f"Heatmap count data CSV saved for {molecule}-{disease}.")

        # Generate and save heatmap
        generate_heatmap(heatmap_csv_path, molecule, disease, output_condition_path)

        # One-hot encode categorical data for PyTorch processing
        merged_encoded = pd.get_dummies(merged_data.drop(['SMILES'], axis=1), columns=['plant', 'molecules', 'Dataset_Name'])
        logging.debug(f"Encoded merged data shape: {merged_encoded.shape}")
        logging.debug(f"Encoded merged data columns: {merged_encoded.columns}")

        # Ensure Cluster is numeric and present
        if 'Cluster' not in merged_encoded.columns:
            logging.error("Cluster column not found after encoding.")
            return

        # Convert all columns to numeric to avoid type errors
        merged_encoded = merged_encoded.apply(pd.to_numeric, errors='coerce')
        if merged_encoded.isnull().values.any():
            logging.error("There are non-numeric values after conversion to numeric types.")
            return

        # Convert DataFrame to tensor for PyTorch processing
        features = torch.tensor(merged_encoded.drop(['Cluster'], axis=1).values, dtype=torch.float32)
        labels = torch.tensor(merged_encoded['Cluster'].values, dtype=torch.long)
        logging.debug(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

        # Prepare DataLoader
        dataset = TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Initialize and train the model
        model = NeuralNetwork(features.shape[1])
        train_model(model, loader, epochs=10)

        # Print confirmation message
        print(f"Processing complete for {molecule}-{disease}. Output saved at: {output_condition_path}")
    else:
        logging.error(f"CSV file not found: {second_csv_path}")

# Main execution (can create our own folder in google drive)
base_directory = '/content/drive/My Drive/AB_SMILES_Dataset'
first_csv_path = '/content/drive/My Drive/AB_SMILES_Dataset/input/total_SMILES_clustering_results.csv'
output_base_directory = '/content/drive/My Drive/AB_SMILES_Dataset/today'

if os.path.exists(first_csv_path):
    first_csv = pd.read_csv(first_csv_path)
    for molecule in ['beta-Sitosterol', 'Palmitic acid', 'Oleic acid', 'Stearic acid', 'Linoleic acid']:
        for disease in [
            'Abdominal pain', 'Analgesics', 'Anemia', 'Anorexia', 'Anthelmintics',
            'Anti-bacterial agents', 'Anti-inflammatory agents', 'Anticonvulsants',
            'Antifungal agents', 'Antineoplastic agents', 'Antioxidants', 'Antipyretics',
            'Antirheumatic agents', 'Aphrodisiacs', 'Appetite stimulants', 'Asthma',
            'Astringents', 'Brain diseases', 'Bronchitis', 'Cardiotonic agents', 'Chest pain',
            'Cholera', 'Colic', 'Common cold', 'Constipation', 'Cooling effect on body',
            'Cough', 'Demulcents', 'Diabetes mellitus', 'Diarrhea', 'Digestive system diseases',
            'Diuretics', 'Dysentery', 'Dyspepsia', 'Dysuria', 'Edema', 'Emollients',
            'Endophthalmitis', 'Expectorants', 'Fever', 'Flatulence', 'Furunculosis',
            'Galactogogues', 'Gastrointestinal diseases', 'General tonic for rejuvenation',
            'Gonorrhea', 'Gout', 'Hair loss', 'Headache', 'Heart diseases', 'Helminthiasis',
            'Hemorrhage', 'Hemorrhoids', 'Hypertension', 'Inflammation', 'Jaundice',
            'Kidney calculi', 'Labor pain', 'Laxatives', 'Leprosy', 'Leukorrhea', 'Liver diseases',
            'Malaria', 'Metabolism', 'Nervous system diseases', 'Pain', 'Paralysis',
            'Parasympatholytics', 'Pharyngitis', 'Postnatal care', 'Rheumatoid arthritis',
            'Scorpion stings', 'Skin diseases', 'Snake bites', 'Splenic diseases',
            'Sprains and strains', 'Stomach diseases', 'Thirst', 'Tuberculosis', 'Ulcer',
            'Urinary bladder calculi', 'Urinary tract infections', 'Urination disorders',
            'Urologic diseases', 'Vomiting', 'Wound healing', 'Wounds and injuries'
        ]:
            process_data(base_directory, molecule, disease, first_csv, output_base_directory)
else:
    logging.error("Failed to load the initial CSV file: " + first_csv_path)
