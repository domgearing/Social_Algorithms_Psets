"""
Mapping between JSON data labels and original Census categories
"""

# sex categories
SEX_MAPPING = {
    "1": "Male",
    "2": "Female"
}

# SCHL educational attainment mapping
SCHL_MAPPING = {
    # Less than high school degree
    "0": "Less than high school degree",
    "01": "Less than high school degree",
    "02": "Less than high school degree",
    "03": "Less than high school degree",
    "04": "Less than high school degree",
    "05": "Less than high school degree",
    "06": "Less than high school degree",
    "07": "Less than high school degree",
    "08": "Less than high school degree",
    "09": "Less than high school degree",
    "10": "Less than high school degree",
    "11": "Less than high school degree",
    "12": "Less than high school degree",
    "13": "Less than high school degree",
    "14": "Less than high school degree",
    "15": "Less than high school degree",
    # High school degree
    "16": "High school degree",
    "17": "High school degree",
    # Some college or Associate degree
    "18": "Some college or Associate degree",
    "19": "Some college or Associate degree",
    "20": "Some college or Associate degree",
    # Bachelor degree
    "21": "Bachelor degree",
    # Graduate degree
    "22": "Graduate degree",
    "23": "Graduate degree",
    "24": "Graduate degree"
}

#Household Income for the Past Twelve Months (Recoded)
HINCP_RC1_MAPPING = {
    "1": "0-24,999",
    "2": "25,000-49,999",
    "3": "50,000-99,999",
    "4": "100,000-149,999",
    "5": "150,000+"
}

#Age Group (Recoded)
AGEP_RC1_MAPPING = {
    "1": "18-29",
    "2": "30-44",
    "3": "45-60",
    "4": "61+"
}

#Census Geography Code
#identifies census region
CENSUS_REGION_MAPPING = {
    "0300000US1": "New England Division",
    "0300000US2": "Middle Atlantic Division",
    "0300000US3": "East North Central Division",
    "0300000US4": "West North Central Division",
    "0300000US5": "South Atlantic Division",
    "0300000US6": "East South Central Division",
    "0300000US7": "West South Central Division",
    "0300000US8": "Mountain Division",
    "0300000US9": "Pacific Division"
}

#Census Geography Code
DATA_LABEL_MAPPING = {
    "SEX": SEX_MAPPING,
    "SCHL": SCHL_MAPPING,
    "HINCP_RC1": HINCP_RC1_MAPPING,
    "AGEP_RC1": AGEP_RC1_MAPPING,
    "ucgid": CENSUS_REGION_MAPPING
}

def get_label_description(field_name: str, code: str) -> str:
    if field_name not in DATA_LABEL_MAPPING:
        return f"Unknown field: {field_name}"
    
    mapping = DATA_LABEL_MAPPING[field_name]
    #try code first, then with leading zeros stripped
    code_str = str(code)
    if code_str in mapping:
        return mapping[code_str]
    
    #try with leading zeros for SCHL codes
    if field_name == 'SCHL' and len(code_str) == 1:
        code_padded = code_str.zfill(2)
        if code_padded in mapping:
            return mapping[code_padded]
    
    return f"Unknown code: {code}"


def describe_record(hincp_rc1: str, agep_rc1: str, sex: str, schl: str, ucgid: str = None) -> dict:
    return {
        'income': get_label_description('HINCP_RC1', hincp_rc1),
        'age_group': get_label_description('AGEP_RC1', agep_rc1),
        'sex': get_label_description('SEX', sex),
        'education': get_label_description('SCHL', schl),
        'region': get_label_description('ucgid', ucgid) if ucgid else None
    }


if __name__ == "__main__":
    # Example usage
    print("=== Data Label Mapping ===\n")
    
    print("SEX Categories:")
    for code, description in SEX_MAPPING.items():
        print(f"  {code}: {description}")
    
    print("\nAGE_Group (AGEP_RC1) Categories:")
    for code, description in AGEP_RC1_MAPPING.items():
        print(f"  {code}: {description}")
    
    print("\nHousehold Income (HINCP_RC1) Categories:")
    for code, description in HINCP_RC1_MAPPING.items():
        print(f"  {code}: {description}")
    
    print("\nEducation (SCHL) Categories (sample):")
    sample_codes = ['0', '16', '19', '21', '22', '23', '24']
    for code in sample_codes:
        print(f"  {code}: {SCHL_MAPPING.get(code, 'N/A')}")
    
    print("\n=== Example Record ===")
    example = describe_record('3', '2', '1', '16', '0300000US1')
    for key, value in example.items():
        print(f"  {key}: {value}")
