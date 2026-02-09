# Define data paths
BASE_DIR = 'D:\ai_brain_tumor_dashboard\src\data'
TRAINING_FOLDER = os.path.join(BASE_DIR, 'Training')
TESTING_FOLDER = os.path.join(BASE_DIR, 'Testing')
VAL_FOLDER = os.path.join(BASE_DIR, 'Validation')
CLASSES = sorted(os.listdir(TRAINING_FOLDER))

# Create validation datset
image_files = glob.glob(os.path.join(TRAINING_FOLDER, '**/*.jpg'), recursive= True)

selected_images = random.sample(image_files, int(len(image_files)*0.2))

#Move the selected images to the validation directory
for image_path in tqdm(selected_images):
    #Get the class name from parent directory
    class_folder = os.path.basename(os.path.dirname(image_path))

    # Create a new folder in the validation directory for the current class
    val_class_path = os.path.join(VAL_FOLDER, class_folder)
    if not os.path.exists(val_class_path):
        os.makedirs(val_class_path)

    # Move the image to the corresponding validation class folder
    val_image_path = os.path.join(val_class_path, os.path.basename(image_path))
    shutil.move(image_path, val_image_path)

folders = [TRAINING_FOLDER, VAL_FOLDER, TESTING_FOLDER]
train_size, val_size, test_size = [len(glob.glob(os.path.join(folder, '**/*.jpg'), recursive = True)) for folder in folders]

print(f'Train_size: {train_size}\nValidation Size: {val_size}\nTest Size: {test_size}')

## EDA
# Pipeline class for the EDA
class EDA:
    def __init__(self, training_folder, testing_folder, valid_folder, classes):
        self.training_folder = training_folder
        self.testing_folder = testing_folder
        self.valid_folder = valid_folder
        self.classes = classes

    def plot_samples(self):
        """
        Function to plot training samples
        """
        fig, ax = plt.subplots(1, 4, figsize=(10, 6))
        for idx, data in enumerate(self._get_training_samples().items()):
            cls, path = data
            img = cv2.imread(path)
            ax[idx].imshow(img)
            ax[idx].set_title(cls)
            ax[idx].axis('off')
        plt.show()

        def plot_counts(self, set_type):
            """
            Function to plot distribution counts.
            :param self: Description
            :param set_type (str): sets to plot counts for. Options: `train` & `set`
            """
            counts = self._get_counts(set_type=set_type)
            X = list(counts.keys())
            y = list(counts.values())

            # Plot counts
            plt.bar(X, y)
            plt.title(f'{set_type.capitalize()} Set Distribution')
            plt.xlabel('Class')
            plt.ylabel('Counts')
            plt.show()

            def _get_training_samples(self):
                """
                Function to retrieve a list of one sample per each class
                """
                paths = [os.path.join(self.training_folder, cls) for cls in self.classes]
                img_paths = {}
                for path, cls in zip(paths, self.classes):
                    img_name = sorted(os.listdir(path))[0]
                    img_paths[cls] = os.path.join(path, img_name)
                return img_paths
            
            def _get_counts(self, set_type):
                """
                Function to get the count of each class in a given set
                
                :param `set_type`: get counts for either `train` and `test`
                return dictionary of counts
                """
                if set_type=='train':
                    folder = self.training_folder
                elif set_type == 'test':
                    folder = self.testing_folder
                else:
                    folder = self.valid_folder

                counts = {}

                #Iterate through class
                for class_type in self.classes:
                    dir = os.path.join(folder, class_type)
                    counts[class_type] = len(os.listdir(dir))
                return counts
            
## Data distribution
# Plot train counts
eda.plot_counts('train')

# Plot validation counts
eda.plot_counts('valid')

# Plot test count
eda.plot_counts('test')

# Plot samples
eda.plot_samples()

## Preprocess data
class Config:
    batch_size = 32
    epochs = 50
    lr = 1e-4
    n_classes = len(CLASSES)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Augment data
train_transform = transform.Compose()        