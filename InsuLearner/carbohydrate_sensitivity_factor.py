__author__ = "Cameron Summers"

"""
This estimates the Carbohydrate Sensitivity Factor (CSF) which is
the concentration level rise of glucose in the blood in mg/dL for
ingestion 1g carbohydrates. It is based on assumptions about
blood volume and carb metabolism.
"""


def estimate_blood_volume_d(height_inches, weight_lbs, gender):
    """
    Estimate blood volume based on height and weight based on Nadler's equation.

    Source: https://pubmed.ncbi.nlm.nih.gov/30252333/

    Args:
        height_inches: height in inches
        weight_lbs: weight in pounds

    Returns:
        blood volume in deciliters (float)
    """
    height_meters = height_inches / 39.37
    weight_kg = weight_lbs / 2.205

    if gender == "male":
        blood_volume_liters = (0.3669 * height_meters**3) + (0.03219 * weight_kg) + 0.6041
    elif gender == "female":
        blood_volume_liters = (0.3561 * height_meters**3) + (0.03308 * weight_kg) + 0.1833
    else:
        raise Exception(f"No equation for gender: {gender}.")

    blood_volume_dl = blood_volume_liters * 10

    return blood_volume_dl


def estimate_csf(height_inches, weight_lbs, gender, metabolism_efficiency_percentage=0.23):
    """
    Estimates the rise in blood glucose concentration based on dissolving
    1000g of glucose into blood.

    23% is a hypothesis:

    In the equations for CIR and ISF from American Association of Endocrinology,
     their CIR and ISF - which is a population average - places CSF average at 3.8 mg/dL / g.
     The blood volume using Nadler equation of an average height and weight person in U.S. is about 55dL.
     100% efficiency would be 1000mg / 55dL = 18.2 mg/dL. So efficiency is 3.8 / 18.2 ~ 21%.

     Corroborating: My 6yo son's blood volume is estimated at about 19.1 dL.
     I estimated his CSF at about 12.5 mg/dL / g.
     100% efficiency for him would be 1000 / 19.1 dL = 52.36 mg/dL.
     So his efficiency is 12.5 / 52.36 = 23.9%.

     Taking a conservative approach (leads to less strong ISF), the default is set to 23%.

     Height and Weights: https://www.cdc.gov/nchs/data/nhsr/nhsr122-508.pdf

    Args:
        height_inches: height in inches
        weight_lbs: weight in pounds
        gender: gender
        metabolism_efficiency_percentage: percentage of carbs metabolized into blood

    Returns:
        Carbohydrate Sensitivity Factor (float)
    """
    blood_volume_dl = estimate_blood_volume_d(height_inches, weight_lbs, gender)
    carbs_mg = 1000
    CSF = carbs_mg / blood_volume_dl * metabolism_efficiency_percentage

    return CSF
