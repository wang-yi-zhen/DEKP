import json
import os
import requests
import time
from io import StringIO
from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

def get_pdb_ids_from_uniprot(uniprot_id):
    url = f'https://www.uniprot.org/uniprot/{uniprot_id}.xml'
    headers = {'User-Agent': 'Mozilla/5.0'}
    max_retries = 3  # Maximum number of retries
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            pdb_ids = []
            if response.status_code == 200:
                from xml.etree import ElementTree as ET
                root = ET.fromstring(response.content)
                namespaces = {'ns': 'http://uniprot.org/uniprot'}
                for db_reference in root.findall(".//ns:dbReference[@type='PDB']", namespaces=namespaces):
                    pdb_id = db_reference.attrib.get('id')
                    if pdb_id:
                        pdb_ids.append(pdb_id)
                return pdb_ids
            else:
                print(f"Request failed, status code: {response.status_code}")
                return []
        except Exception as e:
            print(f"Request exception: {e}, retry count: {attempt + 1}/{max_retries}")
            time.sleep(2)  # Wait for 2 seconds before retrying
            if attempt == max_retries - 1:
                return []

def get_pdb_sequences(pdb_id):
    url = f'https://www.rcsb.org/fasta/entry/{pdb_id}'
    max_retries = 3  # Maximum number of retries
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                sequences = []
                fasta_data = response.text
                fasta_io = StringIO(fasta_data)
                for record in SeqIO.parse(fasta_io, 'fasta'):
                    sequences.append({'chain_id': record.id, 'sequence': str(record.seq)})
                return sequences
            else:
                print(f"Failed to get PDB sequence, status code: {response.status_code}, PDB ID: {pdb_id}")
                return []
        except Exception as e:
            print(f"Request exception: {e}, PDB ID: {pdb_id}, retry count: {attempt + 1}/{max_retries}")
            time.sleep(2)  # Wait for 2 seconds before retrying
            if attempt == max_retries - 1:
                return []

def align_sequences(seq1, seq2):
    alignments = pairwise2.align.globalxx(seq1, seq2)
    if alignments:
        best_alignment = alignments[0]
        score = best_alignment[2]
        aligned_seq1, aligned_seq2 = best_alignment[0], best_alignment[1]
        matches = sum(res1 == res2 for res1, res2 in zip(aligned_seq1, aligned_seq2))
        percentage_identity = matches / max(len(seq1), len(seq2)) * 100
        return score, percentage_identity
    else:
        return 0, 0

def select_best_pdb(uniprot_id, pdb_ids, target_sequence):
    best_score = -1
    best_pdb_id = None
    best_pdb_sequence = None
    best_percentage_identity = 0
    print("Starting sequence alignment...")
    processed_sequences = set()
    for pdb_id in pdb_ids:
        sequences = get_pdb_sequences(pdb_id)
        time.sleep(0.1)
        for seq_info in sequences:
            pdb_sequence = seq_info['sequence']
            if pdb_sequence in processed_sequences:
                continue
            processed_sequences.add(pdb_sequence)
            score, percentage_identity = align_sequences(target_sequence, pdb_sequence)
            print(f"PDB ID: {pdb_id}, Alignment score: {score}, Percentage identity: {percentage_identity:.2f}%")
            if percentage_identity > best_percentage_identity:
                best_score = score
                best_pdb_id = pdb_id
                best_pdb_sequence = pdb_sequence
                best_percentage_identity = percentage_identity
    # Set a threshold for percentage identity, e.g., 90%
    if best_percentage_identity < 90:
        print(f"No PDB structure with sufficient similarity to the target sequence found, UniProt ID: {uniprot_id}")
        return None, None
    print(f"Selected best PDB ID: {best_pdb_id}, Similarity score: {best_score}, Percentage identity: {best_percentage_identity:.2f}%")
    return best_pdb_id, best_pdb_sequence

def download_pdb_file(uniprot_id, pdb_id, save_folder):
    base_url = "https://files.rcsb.org/download/{}.pdb"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    pdb_file_name = f"{uniprot_id}.pdb"
    pdb_file_path = os.path.join(save_folder, pdb_file_name)
    if os.path.exists(pdb_file_path):
        print(f"PDB file already exists, skipping: {pdb_file_name}")
        return True
    url = base_url.format(pdb_id)
    max_retries = 3  # Maximum number of retries
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(pdb_file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"Downloaded PDB file: {pdb_file_name}")
                return True
            else:
                print(f"Download failed, status code: {response.status_code}, PDB ID: {pdb_id}")
                return False
        except Exception as e:
            print(f"Download exception: {e}, PDB ID: {pdb_id}, retry count: {attempt + 1}/{max_retries}")
            time.sleep(2)  # Wait for 2 seconds before retrying
            if attempt == max_retries - 1:
                return False

def download_alphafold_pdb(uniprot_id, save_folder):
    uniprot_id_upper = uniprot_id.upper()
    file_name = f"{uniprot_id}.pdb"
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id_upper}-F1-model_v4.pdb"
    pdb_file_path = os.path.join(save_folder, file_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if os.path.exists(pdb_file_path):
        print(f"AlphaFold PDB file already exists, skipping: {file_name}")
        return True
    max_retries = 3  # Maximum number of retries
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(pdb_file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"Downloaded AlphaFold PDB file: {file_name}")
                return True
            else:
                print(f"AlphaFold PDB file download failed, status code: {response.status_code}, UniProt ID: {uniprot_id}")
                return False
        except Exception as e:
            print(f"Download AlphaFold PDB file exception: {e}, UniProt ID: {uniprot_id}, retry count: {attempt + 1}/{max_retries}")
            time.sleep(2)  # Wait for 2 seconds before retrying
            if attempt == max_retries - 1:
                return False

def process_uniprot_ids(json_file, pdb_folder, alphafold_folder):
    # First, read the JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_uniprot_ids = set()  # Used to store processed UniProt IDs

    # Iterate over each entry
    for idx, entry in enumerate(data):
        uniprot_id = entry.get('UniprotID')
        target_sequence = entry.get('Sequence')

        if not uniprot_id or not target_sequence:
            print(f"Entry {idx + 1}: UniProtID or Sequence field not found")
            entry['PDB_IDs'] = []
            entry['Has_PDB'] = False
            entry['PDB_Source'] = None
            continue

        if uniprot_id in processed_uniprot_ids:
            print(f"UniProt ID {uniprot_id} has been processed, skipping Entry {idx + 1}")
            continue  # Skip already processed UniProt ID

        print(f"Entry {idx + 1}: Processing UniProt ID: {uniprot_id}")

        # Check if the PDB file already exists
        pdb_file_name = f"{uniprot_id}.pdb"
        pdb_file_path_pdb = os.path.join(pdb_folder, pdb_file_name)
        pdb_file_path_alphafold = os.path.join(alphafold_folder, pdb_file_name)

        if os.path.exists(pdb_file_path_pdb):
            print(f"PDB file already exists: {pdb_file_name}, skipping sequence alignment and download.")
            entry['PDB_IDs'] = [uniprot_id]  # Or assign the actual PDB ID
            entry['Has_PDB'] = True
            entry['PDB_Source'] = 'PDB'
            # If needed, read the PDB sequence
            # entry['PDB_Sequence'] = ...
        elif os.path.exists(pdb_file_path_alphafold):
            print(f"AlphaFold PDB file already exists: {pdb_file_name}, skipping sequence alignment and download.")
            entry['PDB_IDs'] = [uniprot_id]
            entry['Has_PDB'] = True
            entry['PDB_Source'] = 'AlphaFold'
            entry['PDB_Sequence'] = target_sequence  # AlphaFold sequence is the same as the UniProt sequence
        else:
            # If the file does not exist, proceed with the normal process
            pdb_ids = get_pdb_ids_from_uniprot(uniprot_id)
            time.sleep(0.2)  # Control request frequency
            if pdb_ids:
                best_pdb_id, best_pdb_sequence = select_best_pdb(uniprot_id, pdb_ids, target_sequence)
                if best_pdb_id:
                    print("Using sequence alignment results, selected the best PDB ID")
                    success = download_pdb_file(uniprot_id, best_pdb_id, pdb_folder)
                    if success:
                        entry['PDB_IDs'] = [best_pdb_id]
                        entry['Has_PDB'] = True
                        entry['PDB_Source'] = 'PDB'
                        entry['PDB_Sequence'] = best_pdb_sequence
                    else:
                        print(f"Download of PDB file failed, PDB ID: {best_pdb_id}")
                        entry['PDB_IDs'] = []
                        entry['Has_PDB'] = False
                        entry['PDB_Source'] = None
                else:
                    print(f"No PDB structure with sufficient similarity to the target sequence found, UniProt ID: {uniprot_id}")
                    # Attempt to download from AlphaFold
                    success = download_alphafold_pdb(uniprot_id, alphafold_folder)
                    if success:
                        entry['PDB_IDs'] = [uniprot_id]
                        entry['Has_PDB'] = True
                        entry['PDB_Source'] = 'AlphaFold'
                        entry['PDB_Sequence'] = target_sequence
                    else:
                        entry['PDB_IDs'] = []
                        entry['Has_PDB'] = False
                        entry['PDB_Source'] = None
            else:
                print(f"No PDB ID corresponding to UniProt ID found: {uniprot_id}")
                # Attempt to download from AlphaFold
                success = download_alphafold_pdb(uniprot_id, alphafold_folder)
                if success:
                    entry['PDB_IDs'] = [uniprot_id]
                    entry['Has_PDB'] = True
                    entry['PDB_Source'] = 'AlphaFold'
                    entry['PDB_Sequence'] = target_sequence
                else:
                    entry['PDB_IDs'] = []
                    entry['Has_PDB'] = False
                    entry['PDB_Source'] = None

        # Update all entries with the same UniProt ID
        for e in data:
            if e.get('UniprotID') == uniprot_id:
                e['PDB_IDs'] = entry.get('PDB_IDs', [])
                e['Has_PDB'] = entry.get('Has_PDB', False)
                e['PDB_Source'] = entry.get('PDB_Source', None)
                e['PDB_Sequence'] = entry.get('PDB_Sequence', None)

        processed_uniprot_ids.add(uniprot_id)  # Add processed UniProt ID to the set

        # After processing each entry, immediately write the updated data back to the JSON file
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Entry {idx + 1}: processing complete, progress saved.")

    print("All entries processed.")

if __name__ == '__main__':
    json_file_path = '/Km_dataset.json'  # Please replace with your JSON file path
    pdb_folder = '/Kmpdbdataset'           # Folder where the PDB database files are saved
    alphafold_folder = '/alphafoldDataset'  # Folder where the AlphaFold files are saved
    process_uniprot_ids(json_file_path, pdb_folder, alphafold_folder)



