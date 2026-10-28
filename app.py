import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import date
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / 'campaign_response_lr_model.pkl')
scaler = joblib.load(BASE_DIR / 'campaign_response_scale.pkl')

FEATURE_COLUMNS = [
    'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
    'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2',
    'age', 'children', 'family_size', 'is_parent', 'customer_tenure',
    'total_spent', 'total_purchases', 'total_campaign_acceptance',
    'avg_spend_per_purchase', 'income_per_person', 'web_purchase_ratio',
    'Education_Basic', 'Education_Graduation', 'Education_Master', 'Education_PhD',
    'Marital_Status_Divorced', 'Marital_Status_Married', 'Marital_Status_Single',
    'Marital_Status_Together', 'Marital_Status_Widow',
]
CLUSTER_COLS = FEATURE_COLUMNS[:29]
REFERENCE_DATE = pd.Timestamp('2014-06-29')

SPEND = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
NUM = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
CAMPAIGNS = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
DIRECT = ['Income', 'Recency'] + SPEND + NUM + CAMPAIGNS

SEG_COLORS = {'Low': '#2f78d6', 'Moderate': '#e29a2d', 'High': '#1fa971'}
FEATURE_NAMES = {
    'total_spent': 'Total spend', 'income_per_person': 'Income per person',
    'web_purchase_ratio': 'Web purchase ratio', 'customer_tenure': 'Customer tenure',
    'avg_spend_per_purchase': 'Avg spend per purchase', 'total_purchases': 'Total purchases',
    'total_campaign_acceptance': 'Past campaign accepts', 'family_size': 'Family size',
    'children': 'Children', 'is_parent': 'Is parent', 'age': 'Age',
}


def engineer(raw):
    d = raw.copy()
    d['Income'] = d['Income'].fillna(d['Income'].median())
    d['Dt_Customer'] = pd.to_datetime(d['Dt_Customer'], format='%d-%m-%Y')
    d['age'] = 2014 - d['Year_Birth']
    d['children'] = d['Kidhome'] + d['Teenhome']
    d['family_size'] = np.where(d['Marital_Status'].isin(['Married', 'Together']),
                                d['children'] + 2, d['children'] + 1)
    d['is_parent'] = (d['children'] > 0).astype(int)
    d['customer_tenure'] = (d['Dt_Customer'].max() - d['Dt_Customer']).dt.days
    d['total_spent'] = d[SPEND].sum(axis=1)
    d['total_purchases'] = d[['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']].sum(axis=1)
    d['total_campaign_acceptance'] = d[CAMPAIGNS].sum(axis=1)
    d['avg_spend_per_purchase'] = d['total_spent'] / (d['total_purchases'] + 1)
    d['income_per_person'] = d['Income'] / d['family_size']
    d['web_purchase_ratio'] = d['NumWebPurchases'] / (d['total_purchases'] + 1)
    return d[d['age'] < 100]


@st.cache_data
def load_data():
    try:
        raw = pd.read_csv(BASE_DIR / 'marketing_campaign.csv', sep='\t')
    except FileNotFoundError:
        return None
    return engineer(raw)


@st.cache_resource
def fit_segments():
    d = load_data()
    if d is None:
        return None
    sc = StandardScaler().fit(d[CLUSTER_COLS])
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(sc.transform(d[CLUSTER_COLS]))
    d = d.assign(cluster=km.labels_)
    order = d.groupby('cluster')['total_spent'].mean().sort_values().index
    names = dict(zip(order, ['Low', 'Moderate', 'High']))
    d['segment'] = d['cluster'].map(names)
    return sc, km, names, d


@st.cache_data
def evaluate():
    d = load_data()
    if d is None:
        return None
    X = pd.get_dummies(d[['Education', 'Marital_Status'] + CLUSTER_COLS],
                       columns=['Education', 'Marital_Status'], drop_first=True)
    X = X[FEATURE_COLUMNS].astype(float)
    _, x_test, _, y_test = train_test_split(X, d['Response'], test_size=0.2,
                                            random_state=42, stratify=d['Response'])
    pred = model.predict(scaler.transform(x_test))
    return accuracy_score(y_test, pred) * 100, recall_score(y_test, pred, pos_label=0) * 100


def make_row(p):
    row = {c: p[c] for c in DIRECT}
    children = p['kids'] + p['teens']
    family = children + (2 if p['marital'] in ('Married', 'Together') else 1)
    spent = sum(p[c] for c in SPEND)
    purchases = p['NumWebPurchases'] + p['NumCatalogPurchases'] + p['NumStorePurchases']
    row.update(
        age=2014 - p['year_birth'], children=children, family_size=family,
        is_parent=int(children > 0),
        customer_tenure=(REFERENCE_DATE - pd.Timestamp(p['enrolled'])).days,
        total_spent=spent, total_purchases=purchases,
        total_campaign_acceptance=sum(p[c] for c in CAMPAIGNS),
        avg_spend_per_purchase=spent / (purchases + 1),
        income_per_person=p['Income'] / family,
        web_purchase_ratio=p['NumWebPurchases'] / (purchases + 1),
    )
    for e in ('Basic', 'Graduation', 'Master', 'PhD'):
        row[f'Education_{e}'] = int(p['education'] == e)
    for m in ('Divorced', 'Married', 'Single', 'Together', 'Widow'):
        row[f'Marital_Status_{m}'] = int(p['marital'] == m)
    return pd.DataFrame([row])[FEATURE_COLUMNS].astype(float)


def customer_segment(row):
    seg = fit_segments()
    if seg is None:
        return None
    sc, km, names, _ = seg
    return names[int(km.predict(sc.transform(row[CLUSTER_COLS]))[0])]


def segment_stats(d):
    share = d['segment'].value_counts(normalize=True) * 100
    resp = d.groupby('segment')['Response'].mean() * 100
    got = d.groupby('segment')['Response'].sum()
    split = (got / got.sum() * 100).round().astype(int)
    split['Low'] = 100 - split['High'] - split['Moderate']
    return share, resp, split


def top_features():
    coef = pd.Series(np.abs(model.coef_[0]), index=FEATURE_COLUMNS)
    top = coef[list(FEATURE_NAMES)].sort_values(ascending=False).head(4)
    return [(FEATURE_NAMES[c], v) for c, v in top.items()]


def donut(pct, color):
    return (
        '<svg viewBox="0 0 120 120" width="110" height="110">'
        '<circle cx="60" cy="60" r="48" fill="none" stroke="#e5e2dc" stroke-width="12"/>'
        f'<circle cx="60" cy="60" r="48" fill="none" stroke="{color}" stroke-width="12" '
        f'stroke-linecap="round" pathLength="100" stroke-dasharray="{max(pct, 0.5):.1f} 100" '
        'transform="rotate(-90 60 60)"/>'
        f'<text x="60" y="68" text-anchor="middle" font-size="24" font-weight="700" fill="#111">{pct:.0f}%</text>'
        '</svg>'
    )


def segment_chart(d, point):
    xmax = d['total_spent'].quantile(0.99) * 1.05
    ymax = d['Income'].quantile(0.99) * 1.05
    X = lambda v: 60 + min(v / xmax, 1) * 520
    Y = lambda v: 330 - min(v / ymax, 1) * 300

    shown = d[(d['total_spent'] <= xmax) & (d['Income'] <= ymax)]
    shown = shown.sample(min(250, len(shown)), random_state=1)
    dots = ''.join(
        f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="5" fill="{SEG_COLORS[g]}"/>'
        for a, b, g in zip(shown['total_spent'], shown['Income'], shown['segment'])
    )
    mark = ''
    if point:
        cx, cy = X(point[0]), Y(point[1])
        right = cx > 440
        mark = (
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="11" fill="none" stroke="#111" stroke-width="2"/>'
            f'<text x="{cx + 8 if right else cx - 8:.1f}" y="{max(cy - 18, 16):.1f}" '
            f'text-anchor="{"end" if right else "start"}" font-size="15" fill="#111">This customer</text>'
        )
    return (
        '<svg viewBox="0 0 600 400" width="100%">'
        '<line x1="60" y1="20" x2="60" y2="330" stroke="#444" stroke-width="1.5"/>'
        '<line x1="60" y1="330" x2="580" y2="330" stroke="#444" stroke-width="1.5"/>'
        f'{dots}{mark}'
        '<text x="18" y="175" transform="rotate(-90 18 175)" text-anchor="middle" font-size="15" fill="#888">Income</text>'
        '<text x="320" y="372" text-anchor="middle" font-size="15" fill="#888">Total spend</text>'
        '</svg>'
    )


st.set_page_config(page_title='Campaign Response Predictor', page_icon='🎯', layout='centered')

st.markdown("""
<style>
.stApp label, .stApp label p, .stApp [data-testid="stWidgetLabel"] { color: #6b6b6b !important; }
.stApp { background: #f0ece4; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 620px; padding-top: 2rem; padding-bottom: 3rem; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff; border: 1px solid #e3dfd6 !important;
    border-radius: 16px; padding: 0.4rem 0.6rem;
}
[data-baseweb="tab-list"] { gap: 8px; border: none; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }
button[data-baseweb="tab"] {
    background: #fff; border: 1px solid #d9d4c9; border-radius: 8px; padding: 4px 14px; height: auto;
}
button[data-baseweb="tab"] p { color: #333 !important; font-size: 0.9rem; }
button[data-baseweb="tab"][aria-selected="true"] { background: #dbe8fb; border-color: #bcd2f3; }
button[data-baseweb="tab"][aria-selected="true"] p { color: #1d4e9e !important; }
.card-title { font-size: 1.05rem; font-weight: 700; color: #111; margin-bottom: 4px; }
.card-sub { font-size: 0.85rem; color: #6b6b6b; margin-bottom: 8px; }
.group { font-size: 0.78rem; font-weight: 700; color: #888; text-transform: uppercase;
         letter-spacing: 0.05em; margin: 14px 0 2px; }
.hdr { display:flex; align-items:center; gap:14px; margin-bottom: 14px; }
.hdr-icon { width:44px; height:44px; border-radius:12px; background:#dbe8fb; color:#1d4e9e;
            display:flex; align-items:center; justify-content:center; font-size:22px; }
.hdr-title { font-size:1.25rem; font-weight:700; color:#111; line-height:1.2; }
.hdr-sub { font-size:0.9rem; color:#555; }
.divider { border-top: 1px solid #e3dfd6; margin: 14px 0; }
.res { display:flex; align-items:center; gap:22px; padding: 4px 6px 8px; }
.res-title { font-size:1.15rem; font-weight:700; color:#111; }
.res-sub { font-size:0.95rem; color:#555; margin: 2px 0 8px; }
.tag { display:inline-block; padding:5px 12px; border-radius:999px; background:#f0ece4;
       color:#333; font-size:0.88rem; }
.tag.green { background:#dff0d4; color:#2f5a1a; }
.tag.amber { background:#fbead6; color:#8a5a0a; }
.legend { display:flex; justify-content:center; gap:30px; margin: 6px 0 14px; font-size:0.95rem; color:#333; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; }
.tiles { display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin: 6px 0; }
.tile { background:#f0ece4; border-radius:12px; padding:12px 14px; }
.tile-label { font-size:0.9rem; color:#555; }
.tile-value { font-size:1.55rem; font-weight:700; color:#111; line-height:1.25; }
.tile-note { font-size:0.9rem; color:#555; }
.drv { display:flex; align-items:center; gap:14px; margin: 9px 0; font-size:0.95rem; color:#222; }
.drv-label { width:150px; }
.drv-track { flex:1; height:10px; background:#f0ece4; border-radius:6px; overflow:hidden; }
.drv-bar { height:100%; border-radius:6px; background:#2f78d6; }
[data-testid="stExpander"] details, [data-testid="stExpander"] details > summary {
    background: #ffffff !important; border-radius: 10px; border-color: #d9d4c9 !important;
}
div.stButton > button {
    width:100%; border-radius:10px; border:1px solid #cfcac0; background:#fff;
    color:#555; padding:0.6rem; font-weight:600;
}
div.stButton > button:hover { border-color:#2f78d6; color:#2f78d6; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hdr"><div class="hdr-icon">◉</div><div>'
    '<div class="hdr-title">Campaign response predictor</div>'
    '<div class="hdr-sub">Predict response, find segments, plan budget</div></div></div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(['1 Predict', '2 Segments', '3 Insights'])

with tab1:
    with st.container(border=True):
        st.markdown('<div class="card-title">1 Predict a customer</div>', unsafe_allow_html=True)

        st.markdown('<div class="group">Customer</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        year_birth = c1.number_input('Year of birth', min_value=1900, max_value=2014, value=1980)
        education = c2.selectbox('Education', ['2n Cycle', 'Basic', 'Graduation', 'Master', 'PhD'])
        marital = c1.selectbox('Marital status', ['Absurd', 'Alone', 'Divorced', 'Married', 'Single',
                                                  'Together', 'Widow', 'YOLO'])
        income = c2.number_input('Income', min_value=0, value=52000, step=1000)
        kids = c1.number_input('Kids at home', min_value=0, max_value=5, value=0)
        teens = c2.number_input('Teens at home', min_value=0, max_value=5, value=0)
        enrolled = c1.date_input('Customer enrollment date', value=date(2013, 6, 15),
                                 min_value=date(2010, 1, 1), max_value=date(2015, 12, 31))
        recency = c2.number_input('Recency (days since last purchase)', min_value=0, value=30)

        st.markdown('<div class="group">Spending</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        wine = c1.number_input('Wine spend', min_value=0, value=200)
        fruit = c2.number_input('Fruit spend', min_value=0, value=20)
        meat = c1.number_input('Meat spend', min_value=0, value=150)
        fish = c2.number_input('Fish spend', min_value=0, value=30)
        sweet = c1.number_input('Sweet spend', min_value=0, value=20)
        gold = c2.number_input('Gold prod spend', min_value=0, value=40)

        st.markdown('<div class="group">Purchases and activity</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        deals = c1.number_input('Num deals purchases', min_value=0, value=2)
        web = c2.number_input('Num web purchases', min_value=0, value=4)
        catalog = c1.number_input('Num catalog purchases', min_value=0, value=2)
        store = c2.number_input('Num store purchases', min_value=0, value=5)
        visits = c1.number_input('Num web visits / month', min_value=0, value=5)

        st.markdown('<div class="group">Past campaigns</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        cmp1 = c1.selectbox('Accepted campaign 1', ['No', 'Yes'])
        cmp2 = c2.selectbox('Accepted campaign 2', ['No', 'Yes'])
        cmp3 = c3.selectbox('Accepted campaign 3', ['No', 'Yes'])
        cmp4 = c1.selectbox('Accepted campaign 4', ['No', 'Yes'])
        cmp5 = c2.selectbox('Accepted campaign 5', ['No', 'Yes'])

        predict = st.button('Predict response')
        out = st.container()

params = dict(
    year_birth=year_birth, education=education, marital=marital, kids=kids, teens=teens,
    enrolled=enrolled, Income=income, Recency=recency,
    MntWines=wine, MntFruits=fruit, MntMeatProducts=meat, MntFishProducts=fish,
    MntSweetProducts=sweet, MntGoldProds=gold,
    NumDealsPurchases=deals, NumWebPurchases=web, NumCatalogPurchases=catalog,
    NumStorePurchases=store, NumWebVisitsMonth=visits,
    AcceptedCmp1=int(cmp1 == 'Yes'), AcceptedCmp2=int(cmp2 == 'Yes'), AcceptedCmp3=int(cmp3 == 'Yes'),
    AcceptedCmp4=int(cmp4 == 'Yes'), AcceptedCmp5=int(cmp5 == 'Yes'),
)

if predict:
    row = make_row(params)
    x = scaler.transform(row)
    st.session_state['result'] = {
        'pred': int(model.predict(x)[0]),
        'prob': float(model.predict_proba(x)[0][1]),
        'segment': customer_segment(row),
        'spent': float(row['total_spent'].iloc[0]),
        'income': float(income),
    }

res = st.session_state.get('result')

with out:
    if res:
        likely = res['pred'] == 1
        seg = res['segment']
        sub = f'Segment: {seg} value' if seg else ''
        tag = ''
        if seg:
            if not likely:
                tag = '<span class="tag">Hold budget</span>'
            elif seg == 'High':
                tag = '<span class="tag green">Priority budget</span>'
            elif seg == 'Moderate':
                tag = '<span class="tag amber">Standard budget</span>'
            else:
                tag = '<span class="tag">Minimal budget</span>'
        st.markdown(
            '<div class="divider"></div><div class="res">'
            + donut(res['prob'] * 100, '#1fa971' if likely else '#d63c3c')
            + '<div><div class="res-title">'
            + ('Likely to respond' if likely else 'Unlikely to respond')
            + f'</div><div class="res-sub">{sub}</div>{tag}</div></div>',
            unsafe_allow_html=True,
        )

with tab2:
    with st.container(border=True):
        st.markdown('<div class="card-title">2 Customer segments (K-Means, K=3)</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Spend vs income, coloured by cluster</div>', unsafe_allow_html=True)
        seg_fit = fit_segments()
        if seg_fit is None:
            st.info('marketing_campaign.csv was not found next to this file, so segments are unavailable.')
        else:
            d = seg_fit[3]
            point = (res['spent'], res['income']) if res else None
            st.markdown(segment_chart(d, point), unsafe_allow_html=True)
            st.markdown(
                '<div class="legend">'
                + ''.join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in SEG_COLORS.items())
                + '</div>',
                unsafe_allow_html=True,
            )
            share, resp, split = segment_stats(d)
            st.markdown(
                '<div class="tiles">'
                '<div class="tile"><div class="tile-label">High segment</div>'
                f'<div class="tile-value">{share["High"]:.0f}% <span class="tile-note">of customers</span></div>'
                f'<div class="tile-note">Response {resp["High"]:.0f}%</div></div>'
                '<div class="tile"><div class="tile-label">Budget split</div>'
                f'<div class="tile-value">{split["High"]} / {split["Moderate"]} / {split["Low"]}</div>'
                '<div class="tile-note">High / Mod / Low</div></div>'
                '</div>',
                unsafe_allow_html=True,
            )
            if not res:
                st.caption('Run a prediction in tab 1 to see where a customer falls.')

with tab3:
    with st.container(border=True):
        st.markdown('<div class="card-title">3 Model insights</div>', unsafe_allow_html=True)
        scores = evaluate()
        if scores:
            acc, rec = scores
            st.markdown(
                '<div class="tiles">'
                f'<div class="tile"><div class="tile-label">Recall (non-responders)</div><div class="tile-value">{rec:.0f}%</div></div>'
                f'<div class="tile"><div class="tile-label">Overall accuracy</div><div class="tile-value">{acc:.0f}%</div></div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info('marketing_campaign.csv was not found, so accuracy and recall are unavailable.')
        st.markdown('<div class="card-sub" style="margin-top:10px">Top engineered features</div>',
                    unsafe_allow_html=True)
        feats = top_features()
        top = feats[0][1] or 1
        st.markdown(''.join(
            f'<div class="drv"><div class="drv-label">{n}</div>'
            f'<div class="drv-track"><div class="drv-bar" style="width:{v / top * 100:.0f}%"></div></div></div>'
            for n, v in feats
        ), unsafe_allow_html=True)
