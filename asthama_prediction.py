{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "fd4105e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "f425d848",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('D:\\Vaibhavi\\Asthama_Dataset.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e8d2bf35",
   "metadata": {
    "scrolled": True
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Patient ID</th>\n",
       "      <th>Age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Coughing</th>\n",
       "      <th>Wheezing</th>\n",
       "      <th>Shortness of breath</th>\n",
       "      <th>Chest Tightning</th>\n",
       "      <th>Rapid Breathing</th>\n",
       "      <th>Fatigue</th>\n",
       "      <th>Trouble Sleeping</th>\n",
       "      <th>Symptoms worsen with activity</th>\n",
       "      <th>Symptoms worsen with allergy</th>\n",
       "      <th>Diagnosis</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75</th>\n",
       "      <td>76</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>76</th>\n",
       "      <td>77</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>77</th>\n",
       "      <td>78</td>\n",
       "      <td>6</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>78</th>\n",
       "      <td>79</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>79</th>\n",
       "      <td>80</td>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>80 rows × 13 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "    Patient ID  Age  Gender  Coughing  Wheezing  Shortness of breath  \\\n",
       "0            1    5       1         1         1                    1   \n",
       "1            2    2       1         1         1                    0   \n",
       "2            3    3       1         0         0                    0   \n",
       "3            4    2       1         1         0                    0   \n",
       "4            5    2       2         1         1                    0   \n",
       "..         ...  ...     ...       ...       ...                  ...   \n",
       "75          76    1       1         0         0                    1   \n",
       "76          77    2       2         0         1                    0   \n",
       "77          78    6       2         1         1                    1   \n",
       "78          79    7       1         1         0                    0   \n",
       "79          80   10       1         0         0                    0   \n",
       "\n",
       "    Chest Tightning  Rapid Breathing  Fatigue  Trouble Sleeping  \\\n",
       "0                 1                1        0                 1   \n",
       "1                 0                1        0                 0   \n",
       "2                 1                0        0                 1   \n",
       "3                 0                1        0                 0   \n",
       "4                 1                1        0                 0   \n",
       "..              ...              ...      ...               ...   \n",
       "75                0                0        1                 0   \n",
       "76                0                0        0                 0   \n",
       "77                0                1        0                 0   \n",
       "78                0                1        0                 1   \n",
       "79                0                1        1                 1   \n",
       "\n",
       "    Symptoms worsen with activity  Symptoms worsen with allergy  Diagnosis  \n",
       "0                               1                             1          1  \n",
       "1                               0                             1          1  \n",
       "2                               0                             0          0  \n",
       "3                               0                             1          1  \n",
       "4                               1                             1          1  \n",
       "..                            ...                           ...        ...  \n",
       "75                              0                             0          0  \n",
       "76                              0                             0          0  \n",
       "77                              1                             0          1  \n",
       "78                              1                             1          1  \n",
       "79                              1                             0          0  \n",
       "\n",
       "[80 rows x 13 columns]"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "162c122d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "id": "7147768e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "9e48c086",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df[['Age','Gender', 'Coughing','Wheezing','Shortness of breath','Chest Tightning','Rapid Breathing','Fatigue','Trouble Sleeping','Symptoms worsen with activity','Symptoms worsen with allergy' ]]\n",
    "y = df['Diagnosis']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "716a69db",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "02aa7ffb",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = GaussianNB()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "ed45c61b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GaussianNB()"
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "84006d30",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "id": "eb3499b5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Enter the following symptoms:\n",
      "Age in years: 16\n",
      "Gender (Male/Female): Female\n",
      "Coughing (Yes/No): No\n",
      "Wheezing (Yes/No): No\n",
      "Shortness of breath (Yes/No): Yes\n",
      "Chest Tightning (Yes/No): Yes\n",
      "Rapid Breathing (Yes/No): No\n",
      "Fatigue (Yes/No): Yes\n",
      "Trouble Sleeping (Yes/No): No\n",
      "Symptoms worsen with activity (Yes/No): No\n",
      "Symptoms worsen with allergy (Yes/No): No\n"
     ]
    }
   ],
   "source": [
    "print(\"Enter the following symptoms:\")\n",
    "age = int(input(\"Age in years: \"))\n",
    "\n",
    "gender = input(\"Gender (Male/Female): \")\n",
    "\n",
    "coughing = input(\"Coughing (Yes/No): \")\n",
    "\n",
    "wheezing = input(\"Wheezing (Yes/No): \")\n",
    "\n",
    "shortness_of_breath = input(\"Shortness of breath (Yes/No): \")\n",
    "\n",
    "chest_tightning = input(\"Chest Tightning (Yes/No): \")\n",
    "\n",
    "rapid_breathing = input(\"Rapid Breathing (Yes/No): \")\n",
    "\n",
    "fatigue = input(\"Fatigue (Yes/No): \")\n",
    "\n",
    "trouble_sleeping = input(\"Trouble Sleeping (Yes/No): \")\n",
    "\n",
    "symptoms_worsen_with_activity= input(\"Symptoms worsen with activity (Yes/No): \")\n",
    "\n",
    "symptoms_worsen_with_allergy= input(\"Symptoms worsen with allergy (Yes/No): \")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "id": "ee4b1804",
   "metadata": {},
   "outputs": [],
   "source": [
    "input_data = pd.DataFrame({\n",
    "    \n",
    "    'Age': [age],\n",
    "    'Gender': [gender],\n",
    "    'Coughing': [coughing],\n",
    "    'Wheezing': [wheezing],\n",
    "    'Shortness of breath': [shortness_of_breath],\n",
    "    'Chest Tightning': [chest_tightning],\n",
    "    'Rapid Breathing': [rapid_breathing],\n",
    "    'Fatigue': [fatigue],\n",
    "    'Trouble Sleeping': [trouble_sleeping],\n",
    "    'Symptoms worsen with activity' :[symptoms_worsen_with_activity],\n",
    "    'Symptoms worsen with allergy' :[symptoms_worsen_with_allergy]\n",
    "})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "id": "71addc0f",
   "metadata": {},
   "outputs": [],
   "source": [
    "input_data = input_data.replace({'Yes': 1, 'No': 0, 'Male':1,'Female':2})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "id": "2cf70bd6",
   "metadata": {},
   "outputs": [],
   "source": [
    "prediction = model.predict(input_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "id": "e6bb2e3a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The predicted diagnosis : No, there is no asthma.\n"
     ]
    }
   ],
   "source": [
    "if prediction == 1:\n",
    "    diagnosis = \"Yes, asthma is present.\"\n",
    "else:\n",
    "    diagnosis = \"No, there is no asthma.\"\n",
    "print(\"The predicted diagnosis :\", diagnosis)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "6fc12e32",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 1.0\n"
     ]
    }
   ],
   "source": [
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print('Accuracy:', accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "id": "44109a60",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Precision: 1.0\n",
      "Recall: 1.0\n",
      "F1-score: 1.0\n"
     ]
    }
   ],
   "source": [
    "precision = precision_score(y_test, y_pred)\n",
    "print('Precision:', precision)\n",
    "\n",
    "recall = recall_score(y_test, y_pred)\n",
    "print('Recall:', recall)\n",
    "\n",
    "f1 = f1_score(y_test, y_pred)\n",
    "print('F1-score:', f1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "54796e1a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
