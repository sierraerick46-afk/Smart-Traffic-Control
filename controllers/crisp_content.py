PHASES = [
    {
        "num": 1,
        "slug": "business-and-data-understanding",
        "title": "Business and data understanding",
        "color": "#2e9e5b",
        "summary": "Define what the traffic lights of Chía need to achieve, which questions the models must answer, and whether the available data is enough to answer them.",
        "intro": "This phase translates the mobility problem of Chía into concrete goals for the machine learning project. It sets the base for everything that follows, so that the technical work stays aligned with what an intersection actually needs.",
        "items": [
            {
                "name": "Business objectives",
                "method": "Translate the needs of the organization into concrete goals for the machine learning project.",
                "applied": [
                    "Reduce the number of vehicles waiting at a signalized intersection compared with a fixed-time plan of 30 seconds per phase.",
                    "Adapt the green time to the type of moment (off-peak, moderate, peak) and to the expected demand of each intersection.",
                    "Offer a web tool where the team can train the models, compare them and watch the control decisions in a simulation.",
                ],
            },
            {
                "name": "Key questions",
                "method": "Identify the specific questions the model has to answer.",
                "applied": [
                    "<b>Group:</b> what kind of moment is the intersection in right now?",
                    "<b>Predict:</b> how many vehicles per hour will arrive at a given intersection, day and hour?",
                    "<b>Decide:</b> should the current green be kept or changed in this second?",
                ],
            },
            {
                "name": "Data assessment",
                "method": "Explore the quality, relevance and sufficiency of the available data.",
                "applied": [
                    "<b>traffic.csv:</b> 48,120 hourly vehicle counts for 4 junctions, from 1 November 2015 to 30 June 2017. Columns: DateTime, Junction, Vehicles, ID.",
                    "<b>Quality:</b> 0 missing values, 0 negative counts and 0 duplicated (date, junction) pairs.",
                    "<b>Balance:</b> junctions 1 to 3 have 14,592 records each; junction 4 only has 4,344, so its models learn from less history.",
                    "<b>Inventory:</b> Red_Semaforica.csv (1,747 signals, 26 columns) gives the structure of a traffic-signal inventory; Intersecciones_Chia.csv holds the 4 intersections of Chía used in the application.",
                ],
            },
            {
                "name": "Feasibility analysis",
                "method": "Consider limitations and risks to make sure the project is technically and organizationally viable.",
                "applied": [
                    "The vehicle counts come from a public hourly traffic dataset and are assigned to 4 intersections of Chía as a case study; they are not measurements taken in Chía.",
                    "There is no access to a real traffic controller, so the decisions are tested in a simulator before anything is proposed for the street.",
                    "Everything runs on CPU (PyTorch CPU build) so the application can be deployed on a free Render instance.",
                    "The hourly granularity is enough to forecast demand, while the second-by-second dynamics are produced by the simulator.",
                ],
            },
        ],
        "tables": [
            {
                "caption": "Average demand per intersection",
                "head": ["Junction", "Intersection", "Records", "Mean veh/h", "P95 veh/h"],
                "rows": [
                    ["1", "Av. Pradilla con Cra 3", "14,592", "45.1", "89"],
                    ["2", "Cra 9 con Calle 10", "14,592", "14.3", "30"],
                    ["3", "Calle 21 (Av. Chilacos) con Cra 9", "14,592", "13.7", "30"],
                    ["4", "Calle 11 con Cra 17", "4,344", "7.3", "13"],
                ],
            }
        ],
        "note": "According to Schmid, Wurst and Wirth (2020), this stage sets the foundation for the rest of the development and keeps the technical effort aligned with real needs.",
    },
    {
        "num": 2,
        "slug": "data-engineering",
        "title": "Data engineering",
        "color": "#3a8fb7",
        "summary": "Select, clean, transform and document the data so the three learning stages receive consistent inputs.",
        "intro": "The data loader (models/data_loader.py) runs the same pipeline every time the application starts. Every transformation is part of the code, so the process can be repeated and reviewed.",
        "items": [
            {
                "name": "Selection",
                "method": "Identify the relevant data sources.",
                "applied": [
                    "From traffic.csv: DateTime, Junction and Vehicles.",
                    "From the intersection inventory: ID, DIRECCION (address) and NUMEROFASES (number of phases).",
                    "The loader finds both files automatically by their columns and prefers the Chía inventory when it is present.",
                ],
            },
            {
                "name": "Cleaning",
                "method": "Fix errors and handle missing values.",
                "applied": [
                    "Dates are parsed and rows with an invalid date are removed.",
                    "Vehicle counts are converted to numbers; empty or negative values are dropped.",
                    "Duplicated (date, junction) records are removed and the data is sorted by date.",
                    "Missing addresses become \"No address\" and missing phase counts default to 2.",
                ],
            },
            {
                "name": "Transformation",
                "method": "Create new features and normalize the data.",
                "applied": [
                    "Time features: hour, day of the week, weekend flag and month.",
                    "Cyclic encoding of the hour with sine and cosine, so that 23:00 and 00:00 are close.",
                    "The cluster from phase Group and a one-hot encoding of the intersection are added as features for the forecast.",
                    "Standardization: K-Means works on standardized profiles, the MLP on standardized inputs and outputs, and the DQN state is scaled between 0 and 1.",
                ],
            },
            {
                "name": "Documentation",
                "method": "Record every transformation that was applied.",
                "applied": [
                    "The full pipeline lives in models/data_loader.py and runs the same way on every start.",
                    "The dashboard shows whether real or synthetic data is loaded, the number of records and the number of intersections.",
                    "The table below lists each transformation and its purpose.",
                ],
            },
        ],
        "tables": [
            {
                "caption": "Transformations applied",
                "head": ["Step", "Input", "Output", "Purpose"],
                "rows": [
                    ["Date parsing", "DateTime", "datetime", "Extract time features"],
                    ["Filtering", "Vehicles", "Counts ≥ 0, no empty values", "Remove invalid records"],
                    ["Deduplication", "Date + junction", "One record per hour and junction", "Avoid counting twice"],
                    ["Merge", "Junction + inventory", "Address and phases per junction", "Name each intersection"],
                    ["Time features", "datetime", "hour, dayofweek, is_weekend, month", "Describe the moment"],
                    ["Cyclic hour", "hour", "sin(hour), cos(hour)", "Keep midnight continuous"],
                    ["One-hot", "junction", "4 binary columns", "Tell the model which intersection it is"],
                ],
            }
        ],
        "note": "ML-Ops.org recommends documenting every transformation and decision so that the process can be reproduced and the model can be reviewed or updated later.",
    },
    {
        "num": 3,
        "slug": "model-engineering",
        "title": "Model engineering",
        "color": "#e0a526",
        "summary": "Choose one learning technique per question, train it, validate it on unseen data and tune its configuration.",
        "intro": "Each key question from phase 1 is answered by a different type of learning. The output of each stage is the input of the next one: the cluster feeds the forecast, and both feed the control agent.",
        "items": [
            {
                "name": "Algorithm selection",
                "method": "Choose the techniques that fit the problem and the data.",
                "applied": [
                    "<b>Group (unsupervised):</b> K-Means with 3 clusters, because the moments of the day have no labels and must be discovered.",
                    "<b>Predict (supervised regression):</b> Random Forest, Gradient Boosting and an MLP neural network in PyTorch, compared under the same conditions.",
                    "<b>Decide (reinforcement learning):</b> a Deep Q-Network (DQN), because the best action depends on the queues and is learned by trial and error.",
                ],
            },
            {
                "name": "Training",
                "method": "Apply the prepared data to the selected algorithm.",
                "applied": [
                    "K-Means is fitted on the 168 day-hour slots of the week, once for all intersections and once for each intersection.",
                    "The three forecasting models are trained in parallel on the same 38,496 training rows.",
                    "The MLP learns with mini-batches of 256 rows for 40 epochs, minimizing the mean squared error.",
                    "The DQN agent plays 60 episodes of 200 simulated seconds on the 4 intersections, storing its experience in a replay buffer.",
                ],
            },
            {
                "name": "Validation",
                "method": "Check performance with data not used in training.",
                "applied": [
                    "Forecast: 80/20 split with a fixed seed (42); every model is scored on the same 9,624 test rows it never saw.",
                    "Clusters: silhouette index computed on the standardized profiles.",
                    "Control: the trained agent is evaluated on new simulated episodes and compared with a fixed-time signal facing the same traffic.",
                ],
            },
            {
                "name": "Hyperparameter tuning",
                "method": "Optimize the configuration to improve performance.",
                "applied": [
                    "The values in the table were adjusted by hand, balancing error, training time and the limits of a free CPU server.",
                    "Tree depth and leaf size limit overfitting in the forests; dropout and weight decay do the same in the MLP.",
                    "For the agent, exploration (epsilon) starts at 1.0 and decreases to 0.05 during the first 70% of the episodes.",
                ],
            },
        ],
        "tables": [
            {
                "caption": "Models and hyperparameters",
                "head": ["Stage", "Model", "Main configuration"],
                "rows": [
                    ["Group", "K-Means", "k = 3, n_init = 10, inputs: mean and standard deviation per day-hour"],
                    ["Predict", "Random Forest", "120 trees, max depth 14, min samples per leaf 2"],
                    ["Predict", "Gradient Boosting", "200 iterations, learning rate 0.08, max depth 8"],
                    ["Predict", "MLP (PyTorch)", "10 inputs → 64 → 32 → 1, ReLU, dropout 0.05, Adam lr 0.001, weight decay 1e-5, 40 epochs"],
                    ["Decide", "DQN (PyTorch)", "9 inputs → 64 → 64 → 2 actions, gamma 0.97, batch 64, replay buffer 20,000, target sync every 200 steps"],
                    ["Simulator", "Intersection", "4 approaches, 2 phases, minimum green 8 s, maximum green 60 s, yellow 3 s, queue limit 40"],
                ],
            }
        ],
        "note": "Schmid et al. (2020) highlight the value of including domain knowledge at this stage; here it appears in the simulator rules (minimum and maximum green, yellow time) and in the reward, which penalizes long queues and unnecessary changes.",
    },
    {
        "num": 4,
        "slug": "model-evaluation",
        "title": "Model evaluation",
        "color": "#d64545",
        "summary": "Measure each model with the right metrics, check how well it generalizes, look for risks and review the results.",
        "intro": "This phase decides whether the models are good enough to be used in the application. The figures below come from a reference run with the real data; each new training gives very similar values.",
        "items": [
            {
                "name": "Performance metrics",
                "method": "Select and apply indicators that fit the type of problem.",
                "applied": [
                    "<b>Group:</b> silhouette index (from -1 to 1). Global value 0.60; per intersection 0.61, 0.59, 0.65 and 0.53.",
                    "<b>Predict (regression):</b> MSE chooses the best model, MAE gives the average error in vehicles and R² the share of variation explained.",
                    "<b>Decide:</b> average queue, vehicles served and vehicles lost, compared with the fixed-time signal.",
                ],
            },
            {
                "name": "Cross-validation",
                "method": "Evaluate the ability to generalize with different subsets of the data.",
                "applied": [
                    "The forecast uses hold-out validation: 80% to train and 20% to test, always with the same seed so the three models face exactly the same test set.",
                    "The agent is evaluated on episodes with seeds different from those used in training, on all 4 intersections.",
                    "Pending improvement: k-fold cross-validation (for example, 5 folds) to report the mean and spread of each metric.",
                ],
            },
            {
                "name": "Risk analysis",
                "method": "Identify problems such as overfitting, underfitting or bias.",
                "applied": [
                    "<b>Overfitting:</b> controlled with depth limits, minimum leaf size, dropout and weight decay, and always measured on data the models did not see.",
                    "<b>Underfitting:</b> the first version without the intersection as a feature reached only R² = 0.11; adding it raised R² to 0.68.",
                    "<b>Bias:</b> junction 4 has a third of the data of the others, and the counts do not come from Chía.",
                    "<b>Control:</b> the agent improves the total queue, but at intersection 3 it was slightly worse than the fixed plan in the reference run.",
                ],
            },
            {
                "name": "Expert review",
                "method": "Validate the results with domain specialists.",
                "applied": [
                    "The results were reviewed with the course professor; the comparison table with MSE, MAE and R² for every model was added after that feedback.",
                    "The simulator rules follow common traffic engineering constraints (minimum green, maximum green and yellow time).",
                    "Pending: a review with the mobility office of Chía before any real use.",
                ],
            },
        ],
        "tables": [
            {
                "caption": "Forecast models on the test set (9,624 rows)",
                "head": ["Model", "MSE", "MAE (veh)", "R²", "Training time"],
                "rows": [
                    ["Random Forest", "137.93", "7.87", "0.678", "≈ 4.5 s"],
                    ["Gradient Boosting", "136.38", "7.84", "0.682", "≈ 6.4 s"],
                    ["MLP (PyTorch) ✓", "136.00", "7.83", "0.683", "≈ 9.0 s"],
                ],
            },
            {
                "caption": "Control agent vs fixed-time signal (60 training episodes)",
                "head": ["Policy", "Average queue (veh)", "Vehicles served", "Vehicles lost"],
                "rows": [
                    ["DQN agent", "10.31", "172.3", "0"],
                    ["Fixed time (30 s)", "11.83", "172.0", "0"],
                ],
            },
        ],
        "note": "According to KDnuggets (2025), a rigorous evaluation is essential to trust a model in real conditions. The three forecasting models perform almost the same; the MLP wins by a small margin, and the agent reduces the average queue by about 13% compared with the fixed plan.",
    },
    {
        "num": 5,
        "slug": "model-deployment",
        "title": "Model deployment",
        "color": "#4a63c9",
        "summary": "Put the validated models inside a working web application, document how to use it and define how it is supervised.",
        "intro": "The models are integrated into a Flask application with a layered architecture. The same code runs on a laptop and on Render.",
        "items": [
            {
                "name": "Production deployment",
                "method": "Integrate the validated model into the operating environment, compatible with existing systems and efficient in resources.",
                "applied": [
                    "Layered architecture: models (data and networks), services (one per stage), controllers (web, API and login) and views (HTML, CSS and JavaScript).",
                    "A JSON API connects the pages with the models: training, forecasts, clusters and the live simulation.",
                    "Gunicorn serves the application and render.yaml describes the deployment on Render.",
                    "CPU-only PyTorch and moderate model sizes keep training under a minute on a free instance.",
                ],
            },
            {
                "name": "Documentation and training",
                "method": "Prepare technical and user manuals, and train the people who will use the system.",
                "applied": [
                    "The Home page explains the problem, objectives, methodology and references.",
                    "These CRISP-ML pages document each phase of the project.",
                    "The dashboard explains every metric next to its value, and the simulation shows the decisions in real time.",
                    "The four team members (Erick, Santiago, Ivan and Edward) have their own user to work with the system.",
                ],
            },
            {
                "name": "Protocols",
                "method": "Define procedures for continuous monitoring, incident handling and user feedback.",
                "applied": [
                    "Access control: the dashboard, the simulation and the API require logging in; the Home and these pages are public.",
                    "Passwords are stored as hashes, never in plain text.",
                    "Every request is logged by the server, which helps to find errors.",
                    "The team reports problems and suggestions through the GitHub repository, where each change is reviewed in a pull request.",
                ],
            },
        ],
        "tables": [
            {
                "caption": "Technology stack",
                "head": ["Layer", "Technology"],
                "rows": [
                    ["Backend", "Python, Flask"],
                    ["Data", "pandas, NumPy"],
                    ["Machine learning", "scikit-learn (K-Means, Random Forest, Gradient Boosting)"],
                    ["Deep learning", "PyTorch (MLP and DQN)"],
                    ["Frontend", "HTML, CSS, JavaScript, Bootstrap"],
                    ["Deployment", "Gunicorn, Render"],
                    ["Collaboration", "Git and GitHub (one branch per phase, pull requests)"],
                ],
            }
        ],
        "note": "ML-Ops.org stresses that a deployed model needs solid ways to collect feedback and detect incidents. This is where the model leaves the controlled development environment and starts producing value.",
    },
    {
        "num": 6,
        "slug": "monitoring-and-maintenance",
        "title": "Monitoring and maintenance",
        "color": "#6b7a72",
        "summary": "Keep checking that the models still perform as expected, detect when the traffic changes and retrain them when needed.",
        "intro": "Traffic changes over time: new roads, works or growth of the town alter the patterns the models learned. This phase defines how the system stays reliable.",
        "items": [
            {
                "name": "Continuous supervision",
                "method": "Regularly monitor the performance of the model in production.",
                "applied": [
                    "Every training shows MSE, MAE and R² for the three forecasting models on the dashboard.",
                    "Every agent training reports the reward per episode and the comparison with the fixed-time signal.",
                    "The live simulation shows the average queue, vehicles served and vehicles lost for each intersection.",
                ],
            },
            {
                "name": "Drift detection",
                "method": "Detect early any deterioration caused by changes in the data or the environment.",
                "applied": [
                    "Compare the MAE of new forecasts with the reference value of about 7.8 vehicles; a sustained increase signals drift.",
                    "Compare the average demand per intersection and per hour of new data with the historical profile.",
                    "Check whether the silhouette index of the clusters drops when new data is added.",
                ],
            },
            {
                "name": "Model update",
                "method": "Retrain periodically with new data or adjusted parameters.",
                "applied": [
                    "The \"Train and compare\" and \"Train agent\" buttons retrain the models with the current data.",
                    "New counts can be added to data/raw; the loader picks them up on the next start.",
                    "The best forecasting model is chosen again by MSE after each training, so the application always uses the current winner.",
                ],
            },
            {
                "name": "Risk review",
                "method": "Continuously assess vulnerabilities and adapt quality controls as the context evolves.",
                "applied": [
                    "Review the intersections where the agent does not beat the fixed plan before trusting it there.",
                    "Replace the public counts with real measurements from Chía as soon as they are available.",
                    "Keep the login and change the passwords when the team changes.",
                ],
            },
        ],
        "tables": [],
        "note": "Schmid et al. (2020) describe monitoring and maintenance as a continuous phase that keeps the system reliable over time, so the model keeps providing value as the data and the context evolve.",
    },
]

REFERENCES = [
    "KDnuggets. (2025). <em>The machine learning lifecycle</em>.",
    "ML-Ops.org. (n.d.). <em>CRISP-ML(Q): The ML lifecycle process</em>.",
    "Schmid, S., Wurst, M., &amp; Wirth, R. (2020). <em>Towards CRISP-ML(Q): A machine learning process model with quality assurance methodology</em>.",
]


def get_phase(num):
    for p in PHASES:
        if p["num"] == num:
            return p
    return None
