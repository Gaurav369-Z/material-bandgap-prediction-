from jarvis.db.figshare import data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from jarvis.core.specie import Specie
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

dft_data = data("dft_3d")
df = pd.DataFrame(dft_data)

def features(atoms):
    elements = atoms["elements"]
    z = []
    en = []
    r = []
    
    for i in elements:
        sp = Specie(i)
        z.append(sp.Z)
        en.append(sp.X)
        try:
            r.append(sp.atomic_rad)
        except:
            r.append(0)

    a_z = sum(z)/len(z)
    a_en = sum(en)/len(en)
    en_dif = max(en)-min(en)
    a_r = sum(r)/len(r)
    return pd.Series([a_z,a_en,en_dif,a_r])

df[["a_z","a_en","en_dif","a_r"]] = df["atoms"].apply(features)
df = df[["density","formation_energy_peratom","optb88vdw_bandgap","spg_number","bulk_modulus_kv","ehull","shear_modulus_gv","nat","a_z","a_en","en_dif","a_r"]]

df = df[df["optb88vdw_bandgap"] != "na"]
df = df.replace("na", pd.NA)
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna() 

x = df[["density","formation_energy_peratom","ehull","nat","spg_number","bulk_modulus_kv","shear_modulus_gv","a_z","a_en","en_dif","a_r"]]
y = df["optb88vdw_bandgap"]

x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size=0.2,random_state=42)

RandomF_model = RandomForestRegressor(n_estimators=100,random_state=42)
RandomF_model.fit(x_tr, y_tr)
RF_pred = RandomF_model.predict(x_te)

r2_RF = r2_score(y_te, RF_pred)
mae_RF = mean_absolute_error(y_te, RF_pred)
print("R2 =", r2_RF)
print("MAE =", mae_RF)

d = float(input("Enter density :"))
f_e = float(input("Enter formation energy per atom :"))
e_h = float(input("Enter ehull :"))
n_a = float(input("Enter nat :"))
spg = float(input("Enter sspg_number :"))
b_m = float((input("Enter bulk modulus :")))
s_m = float(input("Enter shear modulus :"))
a_zi = float(input("Enter avg_z :"))
a_eni = float(input("Enter avg_EN :"))
en_difi = float(input("Enter EN_diff :"))
a_ri = float(input("Enter avg_Radius :"))

n_m = [[d,f_e,e_h,n_a,spg,b_m,s_m,a_zi,a_eni,en_difi,a_ri]]
RF_pred = RandomF_model.predict(n_m)

print("predicted band gap :", RF_pred)



