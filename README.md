# 🌾 Kyield: Predictive Analysis for Indian Agriculture

This project is my take on figuring out how we can use data to make farming a bit more predictable. It follows the **CRISP-DM** lifecycle to answer two big questions: 
1. **Can we predict Rice and Wheat yields 3 months before harvest?** (Spoiler: Yes, we hit around 90% accuracy).
2. **Which crops handle drought better?** (Hint: Rice is surprisingly tough).

---

## 📂 What's in the box?

- **`notebooks/`**: The heart of the analysis. Six phases of step-by-step logic, plus a `Final_Pipeline.ipynb` that runs the whole thing in one go.
- **`src/`**: Reusable Python scripts for cleaning data, scraping Wikipedia, and building the models. No messy copy-pasting here!
- **`app/`**: A modern **FastAPI** web app with a "glassmorphism" UI so you can play with the model yourself.
- **`models/`**: The brain of the operation—saved `.joblib` files for the trained Random Forest and the data scaler.
- **`figures/`**: All the dashboards and plots I generated.
- **`reports/`**: The final deep-dive report (around 1500 words) summarizing the whole journey.

---

## 📊 The Data

I used two main sources for this:
1. **Kaggle**: The [Agricultural Crop Yield in Indian States](https://www.kaggle.com/datasets/srinivas1/agricuture-crop-prediction) dataset (19,689 records).
2. **Wikipedia**: Scraped live data to map states to their geographic zones for better modeling.

---

## 🚀 Quick Start (Let's get it running)

If you have Python installed, just follow these simple steps to get the environment ready:

### 1. Setup the environment
```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

### 2. Run the Web App (The Cool Part)
I built a real-time predictor so you don't have to scroll through code to see it work.
```bash
uvicorn app.main:app --reload
```
Once it's running, open **[http://localhost:8000](http://localhost:8000)** in your browser and start simulating!

### 3. Run the Analysis
If you want to see the "how" behind the numbers:
- Open `notebooks/Final_Pipeline.ipynb` and run all cells.
- It will re-clean the data, re-scrape the zones, and re-train the models from scratch.

---

## 🤖 What does it do?
The code doesn't just "guess." It takes inputs like Rainfall, Fertilizer, and Pesticide, applies a tuned **Random Forest** model, and gives you a forecast in tonnes per hectare. 

It also has a dedicated "Resilience Engine" that audits how crops behave during low-rainfall years, giving us a statistical answer for food security planning. 

---

### 💡 Tech Stack
- **Python** (Pandas, Scikit-learn, BeautifulSoup)
- **FastAPI** (Backend)
- **Vanilla CSS** (Frontend - Inter UI & Glassmorphism)
- **Git** (Version Control)

*Built as part of the KH5004CMD Data Science Coursework.*

---

## 📚 References & Resources

I relied on these technical guides and community discussions to handle the modeling, scraping, and web development parts of the project:

- **Machine Learning**: [Random Forest Regression in Python](https://www.geeksforgeeks.org/random-forest-regression-in-python/) — helped with the tuning logic for the predictor.
- **Data Visualization**: [Correlation Heatmaps with Seaborn](https://www.geeksforgeeks.org/how-to-create-a-seaborn-correlation-heatmap-in-python/) — used this as a reference for the Phase 2 heatmaps.
- **Web Scraping**: [Scraping Wikipedia with BeautifulSoup](https://www.geeksforgeeks.org/implementing-web-scraping-python-beautifulsoup/) — used for the Zone-mapping logic in Phase 3.
- **Web Deployment**: [Serving Jinja2 Templates with FastAPI](https://stackoverflow.com/questions/64516139/how-to-serve-jinja2-templates-with-fastapi-correctly) — helped iron out the frontend-backend connection.
- **Methodology**: [CRISP-DM Data Mining Framework](https://www.geeksforgeeks.org/crisp-dm-data-mining-framework/) — the standard I followed for the project phases.
- **Kaggle**: [Indian Agriculture / Crop Yield Dataset](https://www.kaggle.com/datasets/srinivas1/agricuture-crop-prediction) — the primary source for the historical yield and climate data.
- **Wikipedia**: [States and Union Territories of India](https://en.wikipedia.org/wiki/States_and_union_territories_of_India) — used for mapping states to geographic zones during data augmentation.
- **Library Documentation**: [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — reference for model implementation and cross-validation strategies.
