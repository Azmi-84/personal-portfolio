### Decisions

- Features:
    - No of numerical features: 9
    - Numerical features: 'sonication duration', 'sonication temperature', 'curing duration', 'curing temperature', 'post-curing duration', 'post-curing temperature', 'number of fiber layers', 'nanoparticle weight fraction', 'nanoparticle size'
    - The curing and post-curing temperatures considered in this study refer to the actual processing temperatures applied to the composite laminates during fabrication as reported in each publication.
    - No of categorical features: 4
    - Categorical features: 'manufacturing process', 'nano-particle type', 'fiber orientation', 'test standard'

- Data preprocessing:
    - Missing nanoparticle size information was obtained from manufacturer specifications when referenced in the source articles.
    - Temperature-related fields listed only as “room temperature” were replaced with the representative room temperature of the country where the study was conducted.
    - Rows for which no reasonable inference could be made were removed to maintain dataset consistency.
    - Categorical features were converted into numerical format using scikit-learn's **LabelEncoder**.
    - All numercial features were normalized using **MinMaxScaler**.

- Dataset characteristics and distribution analysis
    - Eight  manufacturing methods were represented: Hand Layup, Vacuum-Assisted Resin Transfer Molding (VARTM), Dip-Coating, Vacuum Bagging, Hot Pressing, Dispersion, Filament Winding, and Plasma Treatment.
