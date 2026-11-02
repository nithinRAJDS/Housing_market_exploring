import os, io, json, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")
st.set_page_config(page_title="EstateIQ | Property Investment Intelligence", page_icon="🏙️", layout="wide", initial_sidebar_state="expanded")

# ---------- Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:linear-gradient(180deg,#f7f9fc 0%,#eef3f8 100%);}
h1,h2,h3 {font-family:'Manrope',sans-serif;letter-spacing:-.035em;color:#14233b;}
.hero {padding:2.0rem 2.2rem;border-radius:22px;background:linear-gradient(115deg,rgba(10,24,45,.96),rgba(17,75,91,.88)),url('https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?auto=format&fit=crop&w=1800&q=85');background-size:cover;color:white;margin-bottom:1.1rem;}
.hero h1 {color:white;font-size:2.45rem;margin:0;}
.hero p {color:#d6e4ef;font-size:1.03rem;max-width:780px;}
.eyebrow {text-transform:uppercase;letter-spacing:.16em;font-size:.72rem;color:#74d7c3;font-weight:800;}
.metric-card {background:white;border:1px solid #e5ebf2;border-radius:16px;padding:16px 18px;box-shadow:0 5px 18px rgba(25,45,75,.045);}
.metric-label {font-size:.78rem;color:#718096;font-weight:700;text-transform:uppercase;letter-spacing:.07em;}
.metric-value {font-family:'Manrope';font-size:1.65rem;color:#14233b;font-weight:800;margin-top:5px;}
.metric-note {font-size:.78rem;color:#7d8998;margin-top:3px;}
.section-kicker {color:#0c8b83;font-size:.73rem;text-transform:uppercase;letter-spacing:.14em;font-weight:800;}
div[data-testid="stMetric"] {background:white;padding:14px 16px;border:1px solid #e5ebf2;border-radius:14px;}
div.stButton>button {border-radius:10px;font-weight:700;}
.small-note {color:#6b7788;font-size:.82rem;}
div[data-testid="stTabs"] button {font-weight:700;}
</style>
""", unsafe_allow_html=True)

DATA_FILE = os.path.join(os.path.dirname(__file__), "india_housing_prices.csv")
TARGET = "Price_in_Lakhs"
NUMERIC = ["BHK","Size_in_SqFt","Year_Built","Floor_No","Total_Floors","Age_of_Property","Nearby_Schools","Nearby_Hospitals"]
CATEGORICAL = ["State","City","Locality","Property_Type","Furnished_Status","Public_Transport_Accessibility","Parking_Space","Security","Amenities","Facing","Owner_Type","Availability_Status"]
REG_FEATURES = NUMERIC + CATEGORICAL
CLF_FEATURES = NUMERIC + CATEGORICAL + ["Price_in_Lakhs"]
REQUIRED = ["State","City","Locality","Property_Type","BHK","Size_in_SqFt","Price_in_Lakhs","Year_Built","Furnished_Status","Floor_No","Total_Floors","Age_of_Property","Nearby_Schools","Nearby_Hospitals","Public_Transport_Accessibility","Parking_Space","Security","Amenities","Facing","Owner_Type","Availability_Status"]

@st.cache_data(show_spinner="Loading 250,000 property records...")
def load_data():
    d = pd.read_csv(DATA_FILE)
    return d

def money_lakh(v):
    return f"₹{v:,.2f} L"

def money_cr(v):
    return f"₹{v/100:,.2f} Cr"

def clean_data(d):
    d = d.copy()
    for c in NUMERIC + [TARGET]:
        if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[TARGET, "Size_in_SqFt"])
    d = d[(d[TARGET] > 0) & (d["Size_in_SqFt"] > 0)]
    return d

def make_preprocessor(features):
    cats = [c for c in features if c in CATEGORICAL]
    nums = [c for c in features if c not in cats]
    # Ordinal encoding keeps the pipeline compact and handles unseen categories safely.
    return ColumnTransformer([
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cats),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nums)
    ], remainder="drop", verbose_feature_names_out=False)

@st.cache_resource(show_spinner="Training the regression and classification models. First launch may take a minute...")
def train_models(data_fingerprint, raw):
    d = clean_data(raw)
    # A reproducible stratified-size sample keeps cloud startup practical while retaining broad coverage.
    if len(d) > 90000:
        d = d.sample(90000, random_state=42)
    Xr = d[REG_FEATURES]
    yr = d[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(Xr, yr, test_size=.2, random_state=42)
    reg = Pipeline([("prep", make_preprocessor(REG_FEATURES)),
                    ("model", HistGradientBoostingRegressor(max_iter=180, learning_rate=.08, max_leaf_nodes=25,
                                                             l2_regularization=1.0, random_state=42))])
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    reg_metrics = {"r2": r2_score(y_test, pred), "mae": mean_absolute_error(y_test, pred),
                   "rmse": mean_squared_error(y_test, pred)**.5}
    reg_sample = pd.DataFrame({"Actual": y_test.values, "Predicted": pred})

    # Screening label: property is priced at/below its city's median observed price per sq ft.
    # Price_per_SqFt in source is expressed in lakh/sq ft; use supplied field when valid.
    d["__pps"] = pd.to_numeric(d.get("Price_per_SqFt", np.nan), errors="coerce")
    calc_pps = d[TARGET] / d["Size_in_SqFt"]  # lakh per sq ft
    d["__pps"] = d["__pps"].where(d["__pps"] > 0, calc_pps)
    city_medians = d.groupby("City")["__pps"].median().to_dict()
    d["__value_candidate"] = (d["__pps"] <= d["City"].map(city_medians)).astype(int)
    Xc = d[CLF_FEATURES]
    yc = d["__value_candidate"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=.2, random_state=42, stratify=yc)
    clf = Pipeline([("prep", make_preprocessor(CLF_FEATURES)),
                    ("model", HistGradientBoostingClassifier(max_iter=150, learning_rate=.08, max_leaf_nodes=25,
                                                               l2_regularization=1.0, random_state=42))])
    clf.fit(Xc_train, yc_train)
    cp = clf.predict(Xc_test)
    cproba = clf.predict_proba(Xc_test)[:,1]
    clf_metrics = {"accuracy": accuracy_score(yc_test, cp), "precision": precision_score(yc_test, cp, zero_division=0),
                   "recall": recall_score(yc_test, cp, zero_division=0), "f1": f1_score(yc_test, cp, zero_division=0),
                   "matrix": confusion_matrix(yc_test, cp).tolist()}
    clf_sample = pd.DataFrame({"Actual": yc_test.values, "Predicted": cp, "Probability": cproba})
    return reg, clf, reg_metrics, clf_metrics, reg_sample, clf_sample, city_medians

def add_derived(d):
    d = d.copy()
    d["Calculated_Price_per_SqFt_Rs"] = d[TARGET] * 100000 / d["Size_in_SqFt"]
    d["Age_of_Property"] = pd.to_numeric(d["Age_of_Property"], errors="coerce")
    return d

def safe_options(series, fallback):
    vals = sorted(series.dropna().astype(str).unique().tolist())
    return vals if vals else fallback

def input_property(d):
    st.subheader("Property details")
    cities = safe_options(d["City"], ["Bengaluru"])
    states = safe_options(d["State"], ["Karnataka"])
    c1,c2,c3 = st.columns(3)
    state = c1.selectbox("State", states, index=states.index("Karnataka") if "Karnataka" in states else 0)
    city = c2.selectbox("City", cities, index=cities.index("Bengaluru") if "Bengaluru" in cities else 0)
    localities = safe_options(d.loc[d.City==city,"Locality"], safe_options(d["Locality"],["Central"]))
    locality = c3.selectbox("Locality", localities)
    c1,c2,c3 = st.columns(3)
    ptype = c1.selectbox("Property type", safe_options(d.Property_Type,["Apartment"]))
    bhk = c2.number_input("BHK",1,10,2)
    size = c3.number_input("Area (sq ft)",300,20000,1200,step=50)
    c1,c2,c3 = st.columns(3)
    price = c1.number_input("Asking price (₹ lakh)",5.0,10000.0,85.0,step=1.0)
    year = c2.number_input("Year built",1950,2035,2020)
    age = max(0, datetime.now().year-int(year))
    furnished = c3.selectbox("Furnishing",safe_options(d.Furnished_Status,["Unfurnished"]))
    c1,c2,c3 = st.columns(3)
    floor = c1.number_input("Floor number",0,100,3)
    floors = c2.number_input("Total floors",1,120,10)
    transport = c3.selectbox("Public transport",safe_options(d.Public_Transport_Accessibility,["Medium"]))
    c1,c2,c3 = st.columns(3)
    schools = c1.number_input("Nearby schools (count)",0,30,3)
    hospitals = c2.number_input("Nearby hospitals (count)",0,30,2)
    parking = c3.selectbox("Parking",safe_options(d.Parking_Space,["Yes"]))
    c1,c2,c3 = st.columns(3)
    security = c1.selectbox("Security",safe_options(d.Security,["Yes"]))
    facing = c2.selectbox("Facing",safe_options(d.Facing,["East"]))
    owner = c3.selectbox("Owner type",safe_options(d.Owner_Type,["Owner"]))
    c1,c2,c3 = st.columns(3)
    amenities = c1.selectbox("Amenities profile",safe_options(d.Amenities,["Basic"]))
    availability = c2.selectbox("Availability",safe_options(d.Availability_Status,["Ready_to_Move"]))
    c3.caption(f"Calculated property age: {age} years")
    return {"State":state,"City":city,"Locality":locality,"Property_Type":ptype,"BHK":int(bhk),
            "Size_in_SqFt":int(size),"Price_in_Lakhs":float(price),"Year_Built":int(year),"Furnished_Status":furnished,
            "Floor_No":int(floor),"Total_Floors":int(floors),"Age_of_Property":age,"Nearby_Schools":int(schools),
            "Nearby_Hospitals":int(hospitals),"Public_Transport_Accessibility":transport,"Parking_Space":parking,
            "Security":security,"Amenities":amenities,"Facing":facing,"Owner_Type":owner,"Availability_Status":availability}

# ---------- Data ----------
if "properties" not in st.session_state:
    st.session_state.properties = load_data()
if "saved_properties" not in st.session_state:
    st.session_state.saved_properties = []
data = clean_data(st.session_state.properties)
fingerprint = f"{len(data)}-{int(data[TARGET].sum())}-{int(data['Size_in_SqFt'].sum())}"
reg_model, clf_model, reg_metrics, clf_metrics, reg_eval, clf_eval, city_medians = train_models(fingerprint, data)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🏙️ EstateIQ")
    st.caption("Property intelligence • India")
    page = st.radio("NAVIGATION",["Overview","AI Property Valuation","Investment Screening","Market Explorer","Add Property","Model Lab","Data & Exports"],label_visibility="collapsed")
    st.divider()
    st.markdown("**Dataset status**")
    st.metric("Property records",f"{len(data):,}")
    st.metric("Cities covered",f"{data.City.nunique():,}")
    st.caption("Models train on a reproducible sample of up to 90,000 records for cloud performance.")
    st.divider()
    st.caption("For educational research only. Not financial, legal or property due-diligence advice.")

# ---------- Header ----------
st.markdown("""
<div class="hero">
 <div class="eyebrow">REAL ESTATE • MACHINE LEARNING • INDIA</div>
 <h1>Invest with clarity.<br>See beyond the listing.</h1>
 <p>Explore Indian housing data, estimate fair-value ranges, screen property listings and compare markets — powered by regression and classification models.</p>
</div>
""",unsafe_allow_html=True)

# ---------- Pages ----------
if page == "Overview":
    st.markdown('<div class="section-kicker">Market pulse</div>',unsafe_allow_html=True)
    avg = data[TARGET].mean(); med = data[TARGET].median(); pps = (data[TARGET]*100000/data.Size_in_SqFt).median()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Properties analysed",f"{len(data):,}")
    c2.metric("Median asking price",money_lakh(med))
    c3.metric("Average asking price",money_lakh(avg))
    c4.metric("Median price / sq ft",f"₹{pps:,.0f}")
    st.markdown("### Explore the market")
    a,b = st.columns([1.45,1])
    with a:
        city_price=data.groupby("City")[TARGET].median().sort_values(ascending=False).head(15).reset_index()
        fig=px.bar(city_price,x=TARGET,y="City",orientation="h",title="Top 15 cities by median property price",template="plotly_white",color=TARGET,color_continuous_scale="Teal")
        fig.update_layout(height=470,yaxis={"categoryorder":"total ascending"},margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig,use_container_width=True)
    with b:
        typ=data.Property_Type.value_counts().reset_index()
        typ.columns=["Property type","Listings"]
        fig=px.pie(typ,names="Property type",values="Listings",hole=.62,title="Property mix",template="plotly_white")
        fig.update_layout(height=470,margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Price distribution")
    fig=px.histogram(data,x=TARGET,nbins=55,marginal="box",title="Asking price distribution (₹ lakh)",template="plotly_white",color_discrete_sequence=["#12877f"])
    fig.update_layout(height=360)
    st.plotly_chart(fig,use_container_width=True)
    st.info("Start with **AI Property Valuation** to estimate a listing price, or **Investment Screening** to assess whether its price per sq ft is below the city benchmark.")

elif page == "AI Property Valuation":
    st.markdown('<div class="section-kicker">Regression engine</div>',unsafe_allow_html=True)
    st.title("AI property valuation")
    st.write("Enter a property's details to estimate its market asking price from comparable patterns in the dataset.")
    with st.form("valuation_form"):
        prop=input_property(data)
        submitted=st.form_submit_button("✨ Estimate property value",use_container_width=True,type="primary")
    if submitted:
        x=pd.DataFrame([{k:prop[k] for k in REG_FEATURES}])
        estimate=float(reg_model.predict(x)[0])
        low=max(0,estimate-reg_metrics["mae"])
        high=estimate+reg_metrics["mae"]
        asking=prop["Price_in_Lakhs"]
        diff=(asking-estimate)/estimate*100 if estimate else 0
        c1,c2,c3=st.columns(3)
        c1.metric("ML estimated value",money_lakh(estimate))
        c2.metric("Indicative range",f"{money_lakh(low)} – {money_lakh(high)}")
        c3.metric("Asking vs estimate",f"{diff:+.1f}%",delta=f"{'Above' if diff>0 else 'Below'} model estimate")
        st.markdown("#### Valuation interpretation")
        if diff < -10: st.success("The asking price is meaningfully below the model estimate. Validate title, condition, location and comparable transactions before proceeding.")
        elif diff > 10: st.warning("The asking price is meaningfully above the model estimate. Consider negotiation and verify local comparable sales.")
        else: st.info("The asking price is broadly aligned with the model estimate.")
        st.caption(f"Model test MAE: {money_lakh(reg_metrics['mae'])}. The range is a simple error-based guide, not a statistical confidence interval.")
        st.session_state["last_valuation"]={"property":prop,"estimate":estimate,"low":low,"high":high}
        if st.button("Add this property to my workspace"):
            st.session_state.saved_properties.append(prop)
            st.success("Property added to your workspace.")
    else:
        st.markdown("#### What the model considers")
        st.write("Location, property type, size, BHK, age, floor, furnishing, amenities, nearby facilities and availability.")

elif page == "Investment Screening":
    st.markdown('<div class="section-kicker">Classification engine</div>',unsafe_allow_html=True)
    st.title("Investment opportunity screening")
    st.write("The classifier learns whether a listing is priced at or below its city's median price per sq ft. This is a **relative-value screening signal**, not a prediction of future returns.")
    with st.form("screen_form"):
        prop=input_property(data)
        submitted=st.form_submit_button("🔎 Screen this property",use_container_width=True,type="primary")
    if submitted:
        x=pd.DataFrame([{k:prop[k] for k in CLF_FEATURES}])
        probability=float(clf_model.predict_proba(x)[0,1])
        pred=int(probability>=.5)
        city_benchmark=city_medians.get(prop["City"],float(data["Price_in_Lakhs"].div(data["Size_in_SqFt"]).median()))
        subject_pps=prop["Price_in_Lakhs"]/prop["Size_in_SqFt"]
        actual_rs=subject_pps*100000
        benchmark_rs=city_benchmark*100000
        c1,c2,c3=st.columns(3)
        c1.metric("Value-screen probability",f"{probability*100:.1f}%")
        c2.metric("Asking price / sq ft",f"₹{actual_rs:,.0f}")
        c3.metric("City median / sq ft",f"₹{benchmark_rs:,.0f}")
        if pred:
            st.success("Potential value candidate — model sees characteristics associated with at-or-below-city-median pricing.")
        else:
            st.warning("Not flagged as a value candidate — compare alternatives or negotiate.")
        fig=go.Figure(go.Indicator(mode="gauge+number",value=probability*100,title={"text":"Model screening probability"},gauge={"axis":{"range":[0,100]},"bar":{"color":"#11877e"},"steps":[{"range":[0,50],"color":"#edf1f5"},{"range":[50,100],"color":"#d5eee8"}]}))
        fig.update_layout(height=260,margin=dict(l=20,r=20,t=50,b=10))
        st.plotly_chart(fig,use_container_width=True)
        st.caption("The classification label is derived from the dataset's city-level median price per sq ft. It does not measure rental yield, appreciation, liquidity or legal quality.")

elif page == "Market Explorer":
    st.markdown('<div class="section-kicker">Locality intelligence</div>',unsafe_allow_html=True)
    st.title("Market explorer")
    c1,c2,c3=st.columns(3)
    cities=["All"]+safe_options(data.City,[])
    chosen_city=c1.selectbox("City",cities)
    ptypes=["All"]+safe_options(data.Property_Type,[])
    chosen_type=c2.selectbox("Property type",ptypes)
    max_price=float(data[TARGET].quantile(.99))
    price_range=c3.slider("Price range (₹ lakh)",float(data[TARGET].min()),max_price,(float(data[TARGET].min()),max_price))
    f=data.copy()
    if chosen_city!="All": f=f[f.City==chosen_city]
    if chosen_type!="All": f=f[f.Property_Type==chosen_type]
    f=f[f[TARGET].between(*price_range)]
    x1,x2,x3,x4=st.columns(4)
    x1.metric("Matching listings",f"{len(f):,}")
    x2.metric("Median price",money_lakh(f[TARGET].median()) if len(f) else "—")
    x3.metric("Median size",f"{f.Size_in_SqFt.median():,.0f} sq ft" if len(f) else "—")
    x4.metric("Median ₹ / sq ft",f"₹{(f[TARGET]*100000/f.Size_in_SqFt).median():,.0f}" if len(f) else "—")
    if len(f):
        l=f.groupby("Locality").agg(Listings=("ID","count"),Median_Price_L=("Price_in_Lakhs","median"),Median_Area=("Size_in_SqFt","median")).sort_values("Median_Price_L",ascending=False).head(20).reset_index()
        fig=px.bar(l,x="Median_Price_L",y="Locality",orientation="h",color="Listings",title="Localities by median asking price",template="plotly_white",color_continuous_scale="Teal")
        fig.update_layout(height=550,yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(f.head(1000),use_container_width=True,hide_index=True)
    else: st.warning("No records match the selected filters.")

elif page == "Add Property":
    st.markdown('<div class="section-kicker">Live workspace</div>',unsafe_allow_html=True)
    st.title("Add a new property")
    st.write("New entries are added to your active session immediately and flow into the live dashboard, market explorer and exports. They are not written back to the original CSV.")
    with st.form("add_property_form"):
        prop=input_property(data)
        submitted=st.form_submit_button("＋ Add property to live dataset",use_container_width=True,type="primary")
    if submitted:
        prop["Price_per_SqFt"]=prop["Price_in_Lakhs"]/prop["Size_in_SqFt"]
        prop["ID"]=int(pd.to_numeric(data.ID,errors="coerce").max()+1) if "ID" in data else len(data)+1
        st.session_state.properties=pd.concat([st.session_state.properties,pd.DataFrame([prop])],ignore_index=True)
        st.session_state.saved_properties.append(prop)
        st.success("Added successfully. The record is now reflected in session analytics and exports.")
        st.info("The trained models remain fixed to the original training dataset for this session. Use Model Lab → Retrain on current session data to incorporate new records.")
    st.markdown("### Recently added")
    if st.session_state.saved_properties:
        st.dataframe(pd.DataFrame(st.session_state.saved_properties).tail(20),use_container_width=True,hide_index=True)
    else: st.caption("No properties added yet.")

elif page == "Model Lab":
    st.markdown('<div class="section-kicker">Model diagnostics</div>',unsafe_allow_html=True)
    st.title("Machine learning model lab")
    tab1,tab2=st.tabs(["Regression • Price prediction","Classification • Value screening"])
    with tab1:
        a,b,c=st.columns(3)
        a.metric("R²",f"{reg_metrics['r2']:.3f}")
        b.metric("MAE",money_lakh(reg_metrics["mae"]))
        c.metric("RMSE",money_lakh(reg_metrics["rmse"]))
        fig=px.scatter(reg_eval.sample(min(2500,len(reg_eval)),random_state=4),x="Actual",y="Predicted",opacity=.45,trendline="ols",title="Actual vs predicted property price",template="plotly_white")
        fig.add_shape(type="line",x0=0,y0=0,x1=float(reg_eval.Actual.max()),y1=float(reg_eval.Actual.max()),line=dict(dash="dash",color="#e07a5f"))
        fig.update_layout(height=460)
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Regression uses HistGradientBoostingRegressor with ordinal-encoded categorical variables. Test split is held out from the training sample.")
    with tab2:
        a,b,c,dcol=st.columns(4)
        a.metric("Accuracy",f"{clf_metrics['accuracy']:.3f}")
        b.metric("Precision",f"{clf_metrics['precision']:.3f}")
        c.metric("Recall",f"{clf_metrics['recall']:.3f}")
        dcol.metric("F1 score",f"{clf_metrics['f1']:.3f}")
        cm=np.array(clf_metrics["matrix"])
        fig=px.imshow(cm,text_auto=True,labels=dict(x="Predicted",y="Actual",color="Count"),x=["Not candidate","Value candidate"],y=["Not candidate","Value candidate"],color_continuous_scale="Teal",title="Confusion matrix")
        fig.update_layout(height=400)
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Classification target: asking price per sq ft at or below the median for the property's city. This is a dataset-derived proxy label.")
    st.divider()
    if st.button("♻️ Retrain models on current session dataset",type="primary"):
        train_models.clear()
        st.cache_data.clear()
        st.session_state["force_retrain"]=True
        st.rerun()
    st.caption("For deployment cost control, the app trains on up to 90,000 randomly sampled rows. Model scores are dataset-specific and do not guarantee real-world investment outcomes.")

elif page == "Data & Exports":
    st.markdown('<div class="section-kicker">Your research desk</div>',unsafe_allow_html=True)
    st.title("Data & exports")
    st.write("Download the dataset, your filtered research sample, model diagnostics and a compact investment note.")
    c1,c2=st.columns(2)
    with c1:
        st.download_button("⬇️ Download full active dataset (CSV)",data=st.session_state.properties.to_csv(index=False).encode("utf-8"),file_name="estateiq_active_properties.csv",mime="text/csv",use_container_width=True)
    with c2:
        st.download_button("⬇️ Download model metrics (JSON)",data=json.dumps({"regression":reg_metrics,"classification":{k:v for k,v in clf_metrics.items()}},indent=2),file_name="estateiq_model_metrics.json",mime="application/json",use_container_width=True)
    report=pd.DataFrame([{"Model":"Regression","Metric":"R2","Value":reg_metrics["r2"]},{"Model":"Regression","Metric":"MAE (₹ lakh)","Value":reg_metrics["mae"]},{"Model":"Regression","Metric":"RMSE (₹ lakh)","Value":reg_metrics["rmse"]},{"Model":"Classification","Metric":"Accuracy","Value":clf_metrics["accuracy"]},{"Model":"Classification","Metric":"Precision","Value":clf_metrics["precision"]},{"Model":"Classification","Metric":"Recall","Value":clf_metrics["recall"]},{"Model":"Classification","Metric":"F1","Value":clf_metrics["f1"]}])
    st.dataframe(report,use_container_width=True,hide_index=True)
    st.download_button("⬇️ Download model scorecard (CSV)",data=report.to_csv(index=False).encode(),file_name="estateiq_model_scorecard.csv",mime="text/csv")
    st.markdown("### Saved properties")
    if st.session_state.saved_properties:
        saved=pd.DataFrame(st.session_state.saved_properties)
        st.dataframe(saved,use_container_width=True,hide_index=True)
        st.download_button("⬇️ Download saved properties",data=saved.to_csv(index=False).encode(),file_name="my_investment_shortlist.csv",mime="text/csv")
    else:
        st.caption("Save a property from valuation or add one from the Add Property page.")

st.divider()
st.markdown("<div class='small-note'>EstateIQ • Decision-support prototype • Model estimates are not appraisals. Verify title, encumbrances, RERA status, taxes, building condition, comparable transactions, rental demand and financing terms independently.</div>",unsafe_allow_html=True)
