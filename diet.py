"""Diet and lifestyle guidance shown after a prediction (general guidance only)."""

# (emoji, food, why it helps)
EAT_MORE = [
    ("🥬", "Leafy greens", "Spinach, methi and cabbage are very low in carbs and rich in fibre."),
    ("🥒", "Bitter & bottle gourd", "Karela and lauki are traditional, low-calorie choices that suit blood sugar control."),
    ("🥣", "Oats & millets", "Oats, ragi, jowar and bajra release sugar slowly and keep you full."),
    ("🫘", "Dal, chana & rajma", "Plant protein plus fibre prevents sharp sugar spikes after meals."),
    ("🐟", "Fish & eggs", "Lean protein and omega-3 fats support heart health."),
    ("🥜", "Almonds & walnuts", "A small handful gives healthy fats that slow sugar absorption."),
    ("🍎", "Guava, apple, berries", "Low-glycemic fruits with fibre. Eat them whole, never as juice."),
    ("🥛", "Curd & paneer", "Protein-rich and low in sugar. Choose unsweetened curd."),
    ("🌿", "Methi seeds & cinnamon", "Commonly used spices that may help improve sugar control."),
]

LIMIT = [
    ("🍚", "White rice & potatoes", "Keep to a small bowl"),
    ("🍌", "Mango, banana, grapes, chikoo", "Small serving only"),
    ("🫓", "Roti", "2 multigrain rotis per meal at most"),
    ("🍪", "Packaged snacks & biscuits", "Check labels for sugar"),
]

AVOID = [
    ("🍬", "Sugar, jaggery, honey & mithai"),
    ("🥤", "Cold drinks & packaged juices"),
    ("🍞", "Maida: white bread, naan, bakery items"),
    ("🍟", "Deep fried food: samosa, pakora, puri"),
    ("⏰", "Skipping meals, then eating one huge meal"),
]

DAILY_PLAN = [
    ("🌅", "Early morning", "Methi-soaked water or plain water with 5 soaked almonds"),
    ("🍳", "Breakfast", "Vegetable oats or besan chilla with curd, or egg whites with sauteed vegetables"),
    ("🍏", "Mid-morning", "1 guava or apple, or a handful of roasted chana"),
    ("🍛", "Lunch", "1-2 multigrain rotis, 1 bowl dal, a large portion of sabzi, salad and curd"),
    ("🍵", "Evening", "Sugar-free green tea with roasted makhana or sprouts chaat"),
    ("🍲", "Dinner", "Light and early (before 8:30 pm): grilled paneer or fish, sauteed vegetables, 1 roti or soup"),
]

LIFESTYLE = [
    ("🚶", "Walk after meals", "A 10 minute walk after each meal lowers the sugar rise."),
    ("🏃", "Exercise daily", "30-45 minutes of brisk walking, cycling or yoga."),
    ("😴", "Sleep 7-8 hours", "Poor sleep raises blood sugar and hunger."),
    ("💧", "Stay hydrated", "Drink 2.5-3 litres of water through the day."),
]

# Shown when the result is "no diabetes"
PREVENTION = [
    ("🥗", "Follow the plate rule", "Half vegetables, a quarter protein, a quarter whole grains."),
    ("🏃", "Stay active", "30 minutes of exercise, 5 days a week."),
    ("🍬", "Cut hidden sugar", "Avoid soft drinks and limit sweets, juices and maida."),
    ("🩸", "Check yearly", "Get fasting glucose tested once a year, more often if family history exists."),
]


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def glucose_status(glucose: float):
    """(label, level) where level is ok / warn / bad."""
    if glucose >= 126:
        return "Diabetic range", "bad"
    if glucose >= 100:
        return "Pre-diabetic range", "warn"
    return "Normal", "ok"


def bmi_status(bmi: float):
    cat = bmi_category(bmi)
    return cat, ("ok" if cat == "Normal" else "warn" if cat in ("Underweight", "Overweight") else "bad")


def bp_status(bp: float):
    if bp >= 90:
        return "High", "bad"
    if bp >= 80:
        return "Slightly high", "warn"
    return "Normal", "ok"


def personal_notes(glucose, bmi, bp):
    notes = []
    if glucose >= 126:
        notes.append("Your glucose reading is in the diabetic range (126 mg/dL or more).")
    elif glucose >= 100:
        notes.append("Your glucose reading is in the pre-diabetic range (100-125 mg/dL).")
    if bmi >= 25:
        notes.append(f"Your BMI is {bmi:.1f}. Losing even 5% of body weight lowers diabetes risk noticeably.")
    if bp >= 90:
        notes.append("Your diastolic blood pressure is high. Cut down on salt as well.")
    return notes
