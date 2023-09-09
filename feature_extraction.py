import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

def featureExtraction(textFile):
  # Data for all websites
  text_file = open(textFile, 'r', encoding="UTF-8")
  text = text_file.read()
  freq = []
  wft = []

  # Cleaning
  text = text.lower()
  words = text.split("/n")
  # words = [word.strip('.,!;()[]') for word in words]
  # words = [word.replace("'s", '') for word in words]

  # finding unique words
  unique = []
  for word in words:
      if word not in unique:
          unique.append(word)

  # Getting frequency of words
  for word in unique:
    freq.append(words.count(word))

  # Combining frequency and words into 1 table
  for i in range(len(freq)):
    wft.append([freq[i], unique[i]])
  wft.sort()
  wft.reverse()

  # Vectorizing and displaying document-term matrix
  cv = CountVectorizer(max_features=10, stop_words='english')
  count_occurrences = cv.fit_transform(words)
  count_array = count_occurrences.toarray()

  df = pd.DataFrame(data=count_array,columns = cv.get_feature_names())

  print(df)
  

# Category 1: Flowers
featureExtraction("flower1dataset.txt")
featureExtraction("flower2dataset.txt")
featureExtraction("flower3dataset.txt")
featureExtraction("flower4dataset.txt")
featureExtraction("flower5dataset.txt")
featureExtraction("flower6dataset.txt")

# Category 2: Fruits
featureExtraction("fruits1dataset.txt")
featureExtraction("fruits2dataset.txt")
featureExtraction("fruits3dataset.txt")
featureExtraction("fruits4dataset.txt")
featureExtraction("fruits5dataset.txt")
featureExtraction("fruits6dataset.txt")

# Category 3: Stationary (Pens and Pencils)
featureExtraction("stationary1dataset.txt")
featureExtraction("stationary2dataset.txt")
featureExtraction("stationary3dataset.txt")
featureExtraction("stationary4dataset.txt")
featureExtraction("stationary5dataset.txt")
featureExtraction("stationary6dataset.txt")
featureExtraction("stationary7dataset.txt")
featureExtraction("stationary8dataset.txt")
featureExtraction("stationary9dataset.txt")
featureExtraction("stationary10dataset.txt")
