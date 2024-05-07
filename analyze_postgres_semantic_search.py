import csv

# Function to convert CSV file
def convert_csv(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = ['id', 'class_index', 'title', 'description']
        
        with open(output_file, 'w', newline='', encoding='utf-8') as newcsvfile:
            writer = csv.DictWriter(newcsvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()

            # Assigning IDs and writing to new CSV
            for idx, row in enumerate(reader, start=1):
                new_row = {
                    'id': str(idx),
                    'class_index': row['Class Index'],
                    'title': row['Title'],
                    'description': row['Description']
                }
                writer.writerow(new_row)

# Usage example
input_file = 'train.csv'
output_file = 'train_fixed.csv'
convert_csv(input_file, output_file)
