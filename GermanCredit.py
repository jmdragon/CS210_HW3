import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Helper Functions
# -------------------------------
def drop_columns_with_most_none(df, n):
    """
    Drop the n columns that have the highest count of the substring 'none'
    (ties broken by the original left-to-right order).
    """
    none_counts = {}
    for col in df.columns:
        # Convert column values to strings, then count occurrences of "none" (case sensitive)
        count = df[col].astype(str).str.count("none").sum()
        none_counts[col] = count

    # Create list with original indices so that ties are broken in the order of appearance.
    cols_with_info = [(col, idx, none_counts[col]) for idx, col in enumerate(df.columns)]
    # Sort by count descending, then by original index (ascending)
    sorted_cols = sorted(cols_with_info, key=lambda x: (-x[2], x[1]))
    cols_to_drop = [col for col, idx, count in sorted_cols[:n]]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped columns: {cols_to_drop}")
    return df

def map_employment(val, years):
    """
    Map the employment value:
      - If the value is 'unemployed' (case insensitive), return "Unemployed".
      - Otherwise, use the numeric years to assign a category:
          < 2 years      => "Amateur"
          2 <= years < 4 => "Professional"
          4 <= years < 7 => "Experienced"
          >= 7 years     => "Expert"
    """
    # Check if the original string indicates unemployment.
    if isinstance(val, str) and val.strip().lower() == "unemployed":
        return "Unemployed"
    # Otherwise, use the numeric value (if available)
    try:
        years = float(years)
    except (ValueError, TypeError):
        return np.nan
    if years < 2:
        return "Amateur"
    elif 2 <= years < 4:
        return "Professional"
    elif 4 <= years < 7:
        return "Experienced"
    else:  # years >= 7
        return "Expert"

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("GermanCredit.csv")
print("Data loaded. Shape:", df.shape)

# -------------------------------
# Preprocessing (31 pts)
# -------------------------------

# [8 pts] Drop the 3 columns with the highest count of 'none' occurrences.
df = drop_columns_with_most_none(df, 3)

# [4 pts] Remove unnecessary apostrophes (the character ‘) from all values.
# This applies to every cell in the DataFrame.
df = df.replace({"‘": ""}, regex=True)

# [5 pts] Modify checking_status column.
checking_map = {
    "no checking": "No Checking",
    "<0": "Low",
    "0<=X<200": "Medium",
    ">=200": "High"
}
df['checking_status'] = df['checking_status'].map(checking_map).fillna(df['checking_status'])

# [5 pts] Modify savings_status column.
savings_map = {
    "no known savings": "No Savings",
    "<100": "Low",
    "100<=X<500": "Medium",
    "500<=X<1000": "High",
    ">=1000": "High"  # Both last categories become "High"
}
df['savings_status'] = df['savings_status'].map(savings_map).fillna(df['savings_status'])

# [4 pts] Change class column values: 'good' -> '1', 'bad' -> '0'
class_map = {"good": "1", "bad": "0"}
df['class'] = df['class'].map(class_map).fillna(df['class'])

# [5 pts] Process employment:
#   Create a new column for numeric employment years.
def try_convert(x):
    try:
        return float(x)
    except:
        return np.nan

# For rows that are not 'unemployed', try converting to numeric.
df['employment_years'] = df['employment'].apply(lambda x: try_convert(x) if not (isinstance(x, str) and x.strip().lower() == "unemployed") else np.nan)
# Now update the employment column based on the value and numeric years.
df['employment'] = df.apply(lambda row: map_employment(row['employment'], row['employment_years']), axis=1)

# -------------------------------
# Analysis (17 pts)
# -------------------------------

# [5 pts] Crosstab: Count of foreign_worker by class.
print("\nForeign Worker vs. Class (credit):")
foreign_class = pd.crosstab(df['foreign_worker'], df['class'])
print(foreign_class)

# [3 pts] Crosstab: Count of employment by savings_status.
print("\nEmployment vs. Savings Status:")
emp_savings = pd.crosstab(df['employment'], df['savings_status'])
print(emp_savings)

# [4 pts] Average credit_amount of single males with 4 <= employment_years < 7.
# Assume personal_status has strings that include both "male" and "single"
cond = (
    df['personal_status'].astype(str).str.lower().str.contains("male") &
    df['personal_status'].astype(str).str.lower().str.contains("single") &
    (df['employment_years'] >= 4) & (df['employment_years'] < 7)
)
avg_credit_amt = df.loc[cond, 'credit_amount'].mean()
print("\nAverage credit_amount for single males with 4<=years<7 employment:", avg_credit_amt)

# [4 pts] Average credit duration for each job type.
# Assuming the credit duration is in the "duration" column and job type in "job".
avg_duration_by_job = df.groupby('job')['duration'].mean()
print("\nAverage credit duration by job type:")
print(avg_duration_by_job)

# [4 pts] For purpose 'education', determine the most common checking_status and savings_status.
edu_df = df[df['purpose'].astype(str).str.lower() == "education"]
if not edu_df.empty:
    common_checking = edu_df['checking_status'].mode().iloc[0]
    common_savings = edu_df['savings_status'].mode().iloc[0]
    print("\nFor purpose 'education':")
    print("Most common checking status:", common_checking)
    print("Most common savings status:", common_savings)
else:
    print("\nNo records found for purpose 'education'.")

# -------------------------------
# Visualization (24 pts)
# -------------------------------

# [9 pts] Subplots for bar charts:
# Bar chart 1: savings_status (x-axis) vs personal_status counts.
# Bar chart 2: checking_status (x-axis) vs personal_status counts.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Crosstab for savings_status and personal_status and plot.
tab1 = pd.crosstab(df['savings_status'], df['personal_status'])
tab1.plot(kind='bar', ax=ax1)
ax1.set_title("Savings Status vs. Personal Status")
ax1.set_xlabel("Savings Status")
ax1.set_ylabel("Count")

# Crosstab for checking_status and personal_status and plot.
tab2 = pd.crosstab(df['checking_status'], df['personal_status'])
tab2.plot(kind='bar', ax=ax2)
ax2.set_title("Checking Status vs. Personal Status")
ax2.set_xlabel("Checking Status")
ax2.set_ylabel("Count")

plt.tight_layout()
plt.show()

# [9 pts] Bar graph for people having credit_amount > 4000:
# Map property_magnitude (x-axis) to average customer age.
# Here we assume the dataset contains a "property_magnitude" column.
high_credit = df[df['credit_amount'] > 4000]
if 'property_magnitude' in df.columns:
    avg_age_by_property = high_credit.groupby('property_magnitude')['age'].mean()
    plt.figure(figsize=(8, 6))
    avg_age_by_property.plot(kind='bar')
    plt.title("Average Customer Age by Property Magnitude (credit_amount > 4000)")
    plt.xlabel("Property Magnitude")
    plt.ylabel("Average Age")
    plt.tight_layout()
    plt.show()
else:
    print("Column 'property_magnitude' not found in the dataset; skipping property magnitude visualization.")

# [6 pts] For people with "High" savings_status and age above 40,
# plot pie charts for Personal status, Credit history, and Job.
subset = df[(df['savings_status'] == "High") & (df['age'] > 40)]
if not subset.empty:
    # Prepare three subplots for the three pie charts.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Pie chart for Personal status.
    counts_personal = subset['personal_status'].value_counts()
    axes[0].pie(counts_personal, labels=counts_personal.index, autopct='%1.1f%%', startangle=90)
    axes[0].set_title("Personal Status Distribution")
    
    # Pie chart for Credit history.
    counts_credit = subset['credit_history'].value_counts()
    axes[1].pie(counts_credit, labels=counts_credit.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title("Credit History Distribution")
    
    # Pie chart for Job.
    counts_job = subset['job'].value_counts()
    axes[2].pie(counts_job, labels=counts_job.index, autopct='%1.1f%%', startangle=90)
    axes[2].set_title("Job Distribution")
    
    plt.tight_layout()
    plt.show()
else:
    print("No records with 'High' savings_status and age above 40 found; skipping pie charts.")

