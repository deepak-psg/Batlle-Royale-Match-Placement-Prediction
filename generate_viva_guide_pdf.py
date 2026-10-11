"""
Generate Project Review & Viva Preparation Guide PDF
Author: Deepak R (24PD09)
"""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable, PageBreak
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and render total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "Battle Royale Placement Prediction — Review & Viva Guide")
            self.drawRightString(612 - 54, 755, "Deepak R (24PD09)")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 747, 612 - 54, 747)

        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 35, page_str)
        self.drawString(54, 35, "Confidential — Machine Learning Package Review")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 47, 612 - 54, 47)
        self.restoreState()


def build_pdf(filename="Battle_Royale_Placement_Prediction_Viva_Guide.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2563EB'),
        spaceAfter=10
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#475569')
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0F766E'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6
    )

    callout_style = ParagraphStyle(
        'Callout_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor('#1E293B')
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#0F172A')
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#0F172A')
    )

    q_style = ParagraphStyle(
        'Q_Style',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=8,
        spaceAfter=2,
        keepWithNext=True
    )

    a_style = ParagraphStyle(
        'A_Style',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6
    )

    story = []

    # =========================================================================
    # HEADER / TITLE BLOCK
    # =========================================================================
    story.append(Paragraph("Battle Royale Match Placement Prediction", title_style))
    story.append(Paragraph("Project Review & Viva Defense Preparation Guide", subtitle_style))
    
    meta_text = (
        "<b>Candidate:</b> Deepak R &nbsp;|&nbsp; "
        "<b>Roll No:</b> 24PD09 &nbsp;|&nbsp; "
        "<b>Subject:</b> Machine Learning Package &nbsp;|&nbsp; "
        "<b>Repository:</b> github.com/deepak-psg/Batlle-Royale-Match-Placement-Prediction"
    )
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=12))

    # =========================================================================
    # SECTION 1: THE 2-MINUTE REVIEW PITCH
    # =========================================================================
    story.append(Paragraph("1. The 2-Minute Review Opening Pitch (What to Say)", h1_style))
    pitch_text = (
        "<i>\"Good morning/afternoon, professors. In battle royale games like PUBG and BGMI, a player's final placement "
        "percentile depends on an interplay of individual combat, zone movement, squad cohesion, and lobby context. "
        "Existing match scorecards provide raw numbers like kills and damage, but offer zero predictive understanding.<br/><br/>"
        "In this project, I engineered a high-performance supervised regression pipeline across <b>4.45 million records</b>. "
        "Because placement is inherently relative rather than absolute, I used <b>Polars</b> to engineer <b>97 features across "
        "4 families</b>: combat efficiency ratios, match context, team aggregates, and lobby-relative percentile ranks.<br/><br/>"
        "To eliminate teammate data leakage, I enforced an <b>80:20 split strictly by Match ID</b> via <code>GroupShuffleSplit</code> "
        "across 47,963 unique matches. I trained three models progressively: <b>Linear Regression</b> as a baseline (MAE: 0.0533), "
        "<b>Random Forest</b> (MAE: 0.0435), and <b>XGBoost Regressor</b> (MAE: <b>0.0372</b>, R²: <b>0.9718</b>) as the champion model. "
        "Finally, I used <b>TreeSHAP</b> to interpret feature attributions and deployed an interactive <b>Streamlit application</b> "
        "for real-time prediction and dynamic explanation.\"</i>"
    )
    
    pitch_table = Table([[Paragraph(pitch_text, callout_style)]], colWidths=[504])
    pitch_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(pitch_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 2: END-TO-END WORKFLOW
    # =========================================================================
    story.append(Paragraph("2. End-to-End Workflow Architecture", h1_style))
    workflow_steps = [
        "<b>1. Raw Ingestion:</b> <code>train_V2.csv</code> loaded with 4,446,966 rows and 29 features.",
        "<b>2. Data Cleaning:</b> Missing target dropped. 1,726 cheater/glitch rows filtered (0m dist with >0 kills, 100% headshot aimbots, snipes >1km, glitches >50 weapons). Cleaned: <b>4,445,240 rows</b>.",
        "<b>3. Polars Feature Engineering:</b> 97 features generated using Rust-backed parallel window operations (combat ratios, lobby context, squad synergy, rank percentiles). Execution completed in <25 seconds.",
        "<b>4. GroupShuffleSplit (80:20):</b> Matches partitioned by <code>matchId</code> (38,370 train matches, 9,593 test matches) to eliminate teammate leakage.",
        "<b>5. Model Training:</b> Progressive training of Linear Regression, Random Forest, and XGBoost (hist binning + L1 absolute error loss).",
        "<b>6. TreeSHAP & Visualization:</b> Local and global feature attributions computed via Shapley values.",
        "<b>7. Production Deployment:</b> Interactive Streamlit web app (<code>app.py</code>) serving real-time inference and local SHAP charts."
    ]
    for step in workflow_steps:
        story.append(Paragraph(f"• {step}", body_style))
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 3: FILE-BY-FILE DIRECTORY BREAKDOWN
    # =========================================================================
    story.append(Paragraph("3. File-by-File Technical Directory Breakdown", h1_style))
    
    file_data = [
        [Paragraph("File / Directory", table_header), Paragraph("Key Role & Mechanism in Project", table_header)],
        [
            Paragraph("<b>preprocess_pipeline.py</b>", table_cell_bold),
            Paragraph("Executes cleaning (<code>clean_data</code>), 4-family feature engineering (<code>engineer_features</code>) using Polars window functions, and match-level 80:20 split (<code>split_matches</code>). Saves parquet files and match ID arrays.", table_cell)
        ],
        [
            Paragraph("<b>train_models.py</b>", table_cell_bold),
            Paragraph("Trains Linear Regression, Random Forest, and XGBoost progressively. Evaluates on 278,001 test records. Computes TreeSHAP explainer and exports figures to <code>reports/figures/</code>.", table_cell)
        ],
        [
            Paragraph("<b>app.py</b>", table_cell_bold),
            Paragraph("Full-stack Streamlit web application. Contains in-game sliders, tactical presets, real-time multi-model prediction, and live interactive SHAP attribution bar chart.", table_cell)
        ],
        [
            Paragraph("<b>models/</b>", table_cell_bold),
            Paragraph("Contains serialized joblib models (<code>xgboost_model.joblib</code>, <code>random_forest.joblib</code>, <code>linear_regression.joblib</code>), <code>shap_explainer.joblib</code>, feature list, and evaluation metrics.", table_cell)
        ],
        [
            Paragraph("<b>reports/figures/</b>", table_cell_bold),
            Paragraph("Publication-quality evaluation plots: <code>model_comparison.png</code> (MAE/RMSE/R²), <code>feature_importance.png</code> (XGBoost Gain), and <code>shap_summary_plot.png</code> (Beeswarm).", table_cell)
        ],
        [
            Paragraph("<b>requirements.txt</b>", table_cell_bold),
            Paragraph("Pinned dependency specifications (polars, xgboost, scikit-learn, shap, streamlit, pyarrow, matplotlib, seaborn).", table_cell)
        ],
        [
            Paragraph("<b>README.md</b>", table_cell_bold),
            Paragraph("Complete academic documentation with problem statement, methodology, evaluation table, and usage commands.", table_cell)
        ]
    ]
    
    file_table = Table(file_data, colWidths=[140, 364])
    file_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(file_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 4: MACHINE LEARNING CONCEPTS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Core Machine Learning Deep-Dive (Concepts & Theory)", h1_style))
    
    story.append(Paragraph("A. Problem Formulation: Supervised Regression", h2_style))
    story.append(Paragraph(
        "The task is supervised regression targeting <code>winPlacePerc</code> ∈ [0.0, 1.0], where 1.0 represents 1st place and 0.0 represents last. "
        "Regression was chosen over classification because placement is ordinal and continuous. Binning into discrete classes would destroy fine-grained rank differences.",
        body_style
    ))

    story.append(Paragraph("B. Data Leakage Prevention via GroupShuffleSplit", h2_style))
    story.append(Paragraph(
        "<b>The Danger:</b> All teammates in a squad share the exact same final placement. A standard random train/test split would place Player A in the train set and their squadmate Player B in the test set, allowing the model to 'memorize' placement through teammate proxy features.<br/>"
        "<b>The Solution:</b> We strictly split by <code>matchId</code> using <code>GroupShuffleSplit</code>. If a match is in the test set, 100% of its players are isolated in the test set.",
        body_style
    ))

    story.append(Paragraph("C. Feature Engineering: The 4 Families", h2_style))
    feat_family_data = [
        [Paragraph("Feature Family", table_header), Paragraph("Rationale", table_header), Paragraph("Key Features Engineered", table_header)],
        [
            Paragraph("<b>1. Efficiency Ratios</b>", table_cell_bold),
            Paragraph("Separates combat effectiveness from volume/kill-stealing.", table_cell),
            Paragraph("<code>damage_per_kill</code>, <code>headshot_rate</code>, <code>kills_per_walkDistance</code>, <code>items_per_walkDistance</code>", table_cell)
        ],
        [
            Paragraph("<b>2. Match Context</b>", table_cell_bold),
            Paragraph("Accounts for lobby size and game mode variations.", table_cell),
            Paragraph("<code>players_in_match</code>, <code>groups_in_match</code>, <code>group_size</code>, <code>match_type_group</code> (solo/duo/squad)", table_cell)
        ],
        [
            Paragraph("<b>3. Team Aggregates</b>", table_cell_bold),
            Paragraph("Captures squad synergy and collective mobility.", table_cell),
            Paragraph("<code>mean</code>, <code>max</code>, <code>min</code>, <code>sum</code> over <code>[matchId, groupId]</code> for walkDistance, damage, kills, heals, boosts", table_cell)
        ],
        [
            Paragraph("<b>4. Match Percentiles</b>", table_cell_bold),
            Paragraph("Normalizes metrics relative to the specific match lobby.", table_cell),
            Paragraph("<code>col_match_mean_ratio</code>, <code>col_match_max_ratio</code>, <code>col_match_rank_perc</code> (rank within lobby)", table_cell)
        ]
    ]
    feat_table = Table(feat_family_data, colWidths=[110, 150, 244])
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F766E')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("D. Progressive Model Architecture: Baseline to Champion", h2_style))
    story.append(Paragraph(
        "• <b>Linear Regression (Baseline):</b> Establishes benchmark MAE (0.0533). Linear hyperplane fits ranked percentiles well, but misses non-linear survival cliffs.<br/>"
        "• <b>Random Forest (Ensemble / Bagging):</b> Trains 100 decision trees in parallel on bootstrap samples. Reduces variance through averaging and models non-linear zone thresholds (MAE: 0.0435).<br/>"
        "• <b>XGBoost Regressor (Primary / Boosting):</b> Trains sequential trees fitting negative gradients of loss. Uses <code>tree_method='hist'</code> for high-speed histogram binning and <code>objective='reg:absoluteerror'</code> to directly minimize MAE. Champion result: <b>MAE: 0.0372, R²: 0.9718</b>.",
        body_style
    ))

    story.append(Paragraph("E. Explainable AI with TreeSHAP", h2_style))
    story.append(Paragraph(
        "Based on cooperative game theory, SHAP decomposes any prediction into additive Shapley contributions: <code>f(x) = φ₀ + Σ φᵢ</code>. "
        "<b>Tactical Finding:</b> Movement (<code>walkDistance_match_rank_perc</code>) and lobby kill rank (<code>killPlace_team_max</code>) have vastly higher SHAP magnitude than raw kill count. Surviving through rotations dominates aggressive early combat.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 5: MODEL BENCHMARK RESULTS TABLE
    # =========================================================================
    story.append(Paragraph("5. Model Evaluation & Benchmark Results", h1_style))
    story.append(Paragraph("Evaluated on <b>278,001 held-out test match records (3,000 matches)</b> with zero leakage:", body_style))
    
    benchmark_data = [
        [Paragraph("Model Architecture", table_header), Paragraph("Test MAE ⬇", table_header), Paragraph("Test RMSE ⬇", table_header), Paragraph("Test R² Score ⬆", table_header), Paragraph("Training Time", table_header)],
        [Paragraph("<b>Linear Regression (Baseline)</b>", table_cell_bold), Paragraph("0.0533", table_cell), Paragraph("0.0741", table_cell), Paragraph("0.9418", table_cell), Paragraph("8.2s", table_cell)],
        [Paragraph("<b>Random Forest Regressor</b>", table_cell_bold), Paragraph("0.0435", table_cell), Paragraph("0.0612", table_cell), Paragraph("0.9604", table_cell), Paragraph("71.6s", table_cell)],
        [Paragraph("<b>XGBoost Regressor (Champion)</b>", table_cell_bold), Paragraph("<b>0.0372</b>", table_cell_bold), Paragraph("<b>0.0516</b>", table_cell_bold), Paragraph("<b>0.9718</b>", table_cell_bold), Paragraph("116.2s", table_cell)]
    ]
    bench_table = Table(benchmark_data, colWidths=[160, 80, 80, 94, 90])
    bench_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor('#DCFCE7')), # Light green highlight for champion
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(bench_table)
    story.append(Spacer(1, 10))

    # Embed Visual Figures
    if os.path.exists("reports/figures/model_comparison.png") and os.path.exists("reports/figures/feature_importance.png"):
        img_w, img_h = 245, 120
        img1 = Image("reports/figures/model_comparison.png", width=img_w, height=img_h)
        img2 = Image("reports/figures/feature_importance.png", width=img_w, height=img_h)
        img_table = Table([[img1, img2]], colWidths=[252, 252])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 6: TOP 10 VIVA QUESTIONS & MODEL ANSWERS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("6. Top 10 High-Yield Viva / Review Questions & Bulletproof Answers", h1_style))

    viva_qa = [
        (
            "Q1: Why did you split the dataset strictly by matchId instead of standard train_test_split?",
            "In battle royale games, all teammates share the exact same placement percentile (winPlacePerc). "
            "A standard random split places teammates from the same match into both train and test sets, causing severe data leakage. "
            "Using GroupShuffleSplit on matchId guarantees 100% of players from any test match are strictly out-of-sample."
        ),
        (
            "Q2: Why did you frame this as a Regression problem rather than Classification?",
            "Placement percentile is continuous and strictly ordered in [0.0, 1.0]. Converting it into discrete classification bins "
            "(e.g., Top 10, Top 50) would discard rank distance and magnitude information. Regression preserves granular ordinal precision."
        ),
        (
            "Q3: Why did XGBoost outperform Linear Regression and Random Forest?",
            "Linear Regression assumes linear relationships and misses threshold effects. Random Forest trains independent trees via bagging. "
            "XGBoost trains sequential trees that specifically learn the gradient residuals of prior trees. With tree_method='hist' and "
            "objective='reg:absoluteerror', XGBoost directly optimized MAE loss, achieving our best score of 0.0372 MAE and 0.9718 R²."
        ),
        (
            "Q4: Why did you use Polars instead of Pandas for feature engineering?",
            "With 4.45M rows, executing 40+ group window operations (.over('matchId') and .over(['matchId', 'groupId'])) in Pandas takes 8–12 minutes "
            "and risks Out-Of-Memory crashes on 8GB RAM systems. Polars is written in Rust, leverages Arrow columnar memory, and finished in under 25 seconds."
        ),
        (
            "Q5: How did you clean cheaters and glitches from the dataset?",
            "I filtered 1,726 anomalous records: (1) movement = 0m with >0 kills, (2) 100% headshot rate with ≥10 kills (aimbots), "
            "(3) longest kill distance >1,000m (impossible physics/render distance), and (4) weapons acquired >50 (inventory glitch)."
        ),
        (
            "Q6: What is the significance of the 4 feature families?",
            "Battle royale placement is relative, not absolute. The 4 families capture: (1) Combat efficiency ratios (damage per kill), "
            "(2) Match context (lobby size and team count), (3) Squad synergy (teammate mean and min movement), and (4) Match-relative percentiles (rank within lobby)."
        ),
        (
            "Q7: What is SHAP, and how does it differ from traditional feature importance?",
            "Traditional feature importance (e.g. split count or Gini gain) only offers global ranking and cannot explain directionality or individual predictions. "
            "SHAP calculates additive Shapley values based on cooperative game theory, explaining exactly how each feature pushed an individual prediction up (+) or down (-)."
        ),
        (
            "Q8: What did SHAP reveal about battle royale strategy?",
            "SHAP revealed that walkDistance_match_rank_perc and killPlace_team_max have much higher impact than raw kill count. "
            "Players who prioritize circle rotation, safe positioning, and team cohesion survive significantly longer than aggressive hot-drop rushers who die early."
        ),
        (
            "Q9: How does the Streamlit web application handle real-time inference with 97 features?",
            "The user inputs only key stats (kills, damage, walk distance, boosts, synergy). The build_feature_vector() function dynamically derives the remaining 97 "
            "features using precomputed match benchmarks (benchmark_stats.json), running XGBoost inference and SHAP bar charts in under 20 milliseconds."
        ),
        (
            "Q10: How do you prevent predictions from going below 0 or above 1?",
            "PUBG placement percentiles are physically bounded in [0.0, 1.0]. During inference and evaluation, model outputs are clipped via "
            "np.clip(y_pred, 0.0, 1.0). This guarantees physical validity and eliminates boundary violations."
        )
    ]

    for q, a in viva_qa:
        story.append(Paragraph(q, q_style))
        story.append(Paragraph(f"<b>Answer:</b> {a}", a_style))

    # Checklist block
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=8))
    checklist_text = (
        "<b>Presentation Day Checklist:</b><br/>"
        "1. Keep Streamlit running locally on <code>localhost:8501</code> (<code>py -3.12 -m streamlit run app.py</code>) for a 30-second live demo.<br/>"
        "2. Keep the 3 figures in <code>reports/figures/</code> ready for visual explanation.<br/>"
        "3. Emphasize: (a) 4.45M rows cleaned, (b) GroupShuffleSplit to eliminate leakage, (c) 97 Polars features, (d) XGBoost champion MAE: 0.0372."
    )
    story.append(Paragraph(checklist_text, callout_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}!")


if __name__ == "__main__":
    build_pdf()
