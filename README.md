# Fruit Classification

This project is for classifying different types of fruits using a Convolutional Neural Network (CNN).

## Project Structure

```
fruit-classification/
├── dataset/            # Your scraped images will go here
├── scraper/            # Selenium scraper code
│   └── scrape.py
├── model/              # Your CNN code, training script
│   └── train.py
├── README.md
├── .gitignore
└── requirements.txt
```

- **dataset/**: Contains the images scraped for training/testing.
- **scraper/**: Contains the Selenium-based web scraper code.
- **model/**: Contains the CNN model and training scripts.
- **requirements.txt**: Lists the Python dependencies.
- **.gitignore**: Specifies files and folders to be ignored by git. 