import csv

def log_to_csv(filename, data, headliners):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headliners)  # Write the headers
        writer.writerows(data)  # Write all the rows



def read_csv_to_list(filename, headliners):
    csv_to_list = []
    with open(filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
    
    # Loop through each row in the CSV file
        for row in csv_reader:
            # Use the 'image' as the key and create a dictionary for category and descriptions
            csv_to_list.append([row[headliner] for headliner in headliners])
    return csv_to_list