Cloud eQMS ROI Simulator 📊
Thesis Artifact: Digitalization of Quality Processes in the Pharmaceutical Industry
🎓 Academic Context
Institution: University of Piraeus (UNIPI)
Program: MBA in Total Quality Management (TQM)
Author: Dimitrios Vergis
Supervisor: Dr. Georgios Bohoris
Thesis Title: Digitalization of Quality Processes in the Pharmaceutical Industry through eQMS: A Strategic Roadmap for Cloud SaaS Validation, Compliance, and ROI
💡 Project Overview
This application is a Stochastic Decision Support System developed to quantify the financial Return on Investment (ROI) of transitioning from a legacy (paper/hybrid) Quality Management System to a Cloud-based eQMS.Unlike static Excel models that rely on deterministic averages, this tool utilizes Monte Carlo Simulation ($N=5,000$ iterations) to model the variability and risk inherent in pharmaceutical quality processes.
Key Features
Digital Twin Logic: Simulates "Legacy" vs. "Digital" states for every generated event.
Complexity Driver: Uses a Log-Normal distribution to model event complexity (1–5), accurately reflecting the "long tail" of costly investigations.
Dynamic Resource Allocation: Uses a Poisson distribution to simulate cross-functional departmental involvement based on event complexity.
Econometric Validation: Includes an embedded OLS Regression (with HC3 Robust Standard Errors) to statistically validate the cost savings per event.
🚀 Live Demo
You can access the deployed application directly via Streamlit Cloud:
👉 Launch eQMS ROI Simulator
💻 Local Installation & Usage
To run this tool on your local machine for inspection or development:
Prerequisites
Python 3.9 or higher
pip (Python Package Installer)
Installation StepsClone the repository:Bashgit clone https://github.com/YOUR_USERNAME/eqms-roi-simulator.git
cd eqms-roi-simulator
Install dependencies:Bashpip install -r requirements.txt
Run the application:Bashstreamlit run app.py
⚙️ How It Works (The Math)
The simulation engine operates on the following logic:
Event Generation: The user defines the annual volume for various QMS modules (Deviations, CAPA, Change Control).
Stochastic assignment:
      Complexity ($C$): $C \sim LogNormal(\mu=0, \sigma=0.2)
      $Departments ($D$): $D \sim Poisson(\lambda = 1 + 0.7C)
$Cost Calculation:
      Legacy Cost: $Hours_{Base} + (Penalty_{Linkage} \times Links) + Noise$
      Digital Cost: $Hours_{Base} + (Penalty_{Linkage} \times (1 - SavingsFactor)) + Noise$
Regression Analysis:The tool performs an Ordinary Least Squares (OLS) regression to determine the coefficient $\beta_1$ (System State), which represents the statistically significant saving per event.$$Cost = \beta_0 + \beta_1(State) + \beta_2(Complexity) + \epsilon$$
📂 Repository Structure
    app.py: The main application script containing the simulation engine and UI.
    requirements.txt: List of Python libraries required (pandas, numpy, statsmodels, etc.).
    README.md: Project documentation.
📄 LicenseThis project is open-source and available under the MIT License.
