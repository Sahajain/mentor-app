# CLAT Mentor Recommender

This is a simple tool to help CLAT aspirants find the best mentors based on their study preferences.

---

## How to Use on Your PC


### 1. Download or Clone the Project

Make sure you have these files in one folder:
- `mentor_recommender_app.py`
- `mentors.csv`
- `requirements.txt`

---

### 2. Open the folder in your texteditor and open the terminal

- Install required packages

Run this command:

```bash
pip install -r requirements.txt
```

- Start the app
Run this command:

```bash
streamlit run mentor_recommender_app.py
```
## Summary of the Approach

The CLAT Mentor Recommender system uses a simple but effective AI technique to match students with mentors based on profile similarity.

### Step-by-Step:

1. **Input Collection**  
   - The student fills in their preferences: subjects, learning style, prep level, and target colleges.

2. **Text Feature Creation**  
   - Each mentor profile is converted into a combined text format based on:
     - Subject expertise
     - College
     - Teaching style
     - Experience level  
   - The student’s input is also combined into a similar format.

3. **Vectorization with TF-IDF**  
   - Both mentor profiles and the student profile are converted into numeric vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**, which gives importance to unique terms.

4. **Similarity Matching using Cosine Similarity**  
   - The app calculates the **cosine similarity** between the student vector and each mentor vector.
   - Cosine similarity helps find the mentors whose features are most aligned with the student’s profile.

5. **Top 3 Recommendations**  
   - The system returns the top 3 mentors with the highest similarity scores.




