import csv

def convert_csv(input_file: str, output_file: str) -> None:
    """
    Convert a CSV file with specific headers to a new CSV file with specified field names.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.

    Returns:
        None
    """
    fieldnames = ['id', 'class_index', 'title', 'description']
    
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as newcsvfile:
            writer = csv.DictWriter(newcsvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()

            for idx, row in enumerate(reader, start=1):
                new_row = {
                    'id': str(idx),
                    'class_index': row['Class Index'],
                    'title': row['Title'],
                    'description': row['Description']
                }
                writer.writerow(new_row)

# Usage example
input_file = 'excel_files/train.csv'
output_file = 'excel_files/train_fixed.csv'
convert_csv(input_file, output_file)
