"""
Plate Info Decoder — Tier 1 (offline, zero cost)
=================================================
Decodes everything that can be extracted from just the plate text
(no API, no internet, no database lookup).

Returns 5 pieces of info per plate:
  1. State / Union Territory (full name)
  2. RTO code (e.g. KA-02)
  3. RTO city/region (e.g. "Bangalore Central")
  4. Plate category (Private / Commercial / BH-series / etc.)
  5. Registration series (RTO series + number)

Plus the vehicle type (from YOLO detection: car/bike/truck/bus).

All data bundled in-code. No external files needed.
"""

import re


# =====================================================================
# FULL STATE CODE -> STATE NAME MAPPING
# =====================================================================
STATE_NAMES = {
    'AP': 'Andhra Pradesh',
    'AR': 'Arunachal Pradesh',
    'AS': 'Assam',
    'BR': 'Bihar',
    'CG': 'Chhattisgarh',
    'GA': 'Goa',
    'GJ': 'Gujarat',
    'HR': 'Haryana',
    'HP': 'Himachal Pradesh',
    'JK': 'Jammu & Kashmir',
    'JH': 'Jharkhand',
    'KA': 'Karnataka',
    'KL': 'Kerala',
    'LA': 'Ladakh',
    'LD': 'Lakshadweep',
    'MP': 'Madhya Pradesh',
    'MH': 'Maharashtra',
    'MN': 'Manipur',
    'ML': 'Meghalaya',
    'MZ': 'Mizoram',
    'NL': 'Nagaland',
    'OD': 'Odisha',
    'OR': 'Odisha',  # legacy
    'PB': 'Punjab',
    'RJ': 'Rajasthan',
    'SK': 'Sikkim',
    'TN': 'Tamil Nadu',
    'TS': 'Telangana',
    'TR': 'Tripura',
    'UP': 'Uttar Pradesh',
    'UK': 'Uttarakhand',
    'UA': 'Uttarakhand',  # legacy
    'WB': 'West Bengal',
    # Union Territories
    'AN': 'Andaman & Nicobar Islands',
    'CH': 'Chandigarh',
    'DD': 'Dadra & Nagar Haveli and Daman & Diu',
    'DN': 'Dadra & Nagar Haveli',
    'DL': 'Delhi',
    'PY': 'Puducherry',
    # Special
    'BH': 'Bharat Series (All India)',
}


# =====================================================================
# RTO CODE -> CITY/REGION (abridged but covers major cities)
# =====================================================================
# Format: {state_code: {rto_number: city_name}}
# Focused on major cities where CCTV footage is most likely from.
RTO_CITIES = {
    'KA': {
        '01': 'Bangalore Central', '02': 'Bangalore-Rajajinagar',
        '03': 'Bangalore East', '04': 'Bangalore North',
        '05': 'Bangalore South', '06': 'Tumkur', '07': 'Kolar',
        '08': 'Chikmagalur', '09': 'Mysore', '10': 'Chamarajanagar',
        '11': 'Mandya', '12': 'Madikeri', '13': 'Hassan',
        '14': 'Shimoga', '15': 'Sagar', '16': 'Chitradurga',
        '17': 'Davanagere', '18': 'Chikkanayakanahalli',
        '19': 'Udupi', '20': 'Mangalore', '21': 'Puttur',
        '22': 'Belgaum', '23': 'Chikkodi', '24': 'Bailhongal',
        '25': 'Dharwad', '26': 'Gadag', '27': 'Haveri',
        '28': 'Bijapur', '29': 'Bagalkot', '30': 'Karwar',
        '31': 'Sirsi', '32': 'Gulbarga', '33': 'Yadgir',
        '34': 'Bellary', '35': 'Hospet', '36': 'Raichur',
        '37': 'Koppal', '38': 'Bidar', '39': 'Bhalki',
        '40': 'Chikkaballapur', '41': 'Bangalore-Jayanagar',
        '42': 'Ramanagaram', '43': 'Devanahalli',
        '44': 'Bangalore Electronic City', '45': 'Bangalore-Yelahanka',
        '46': 'Bangalore KR Puram', '47': 'Bangalore-Marathahalli',
        '48': 'Bangalore Indiranagar', '49': 'Bangalore Basavanagudi',
        '50': 'Bangalore Banashankari', '51': 'Bangalore-Yeshwanthpur',
        '52': 'Nelamangala', '53': 'Bangalore HSR', '55': 'Tiptur',
        '57': 'Bangarpet', '58': 'Bangalore',
    },
    'MH': {
        '01': 'Mumbai Central', '02': 'Mumbai West',
        '03': 'Mumbai East', '04': 'Thane', '05': 'Kalyan',
        '06': 'Pen', '07': 'Sindhudurg', '08': 'Ratnagiri',
        '09': 'Kolhapur', '10': 'Sangli', '11': 'Satara',
        '12': 'Pune', '13': 'Solapur', '14': 'Pimpri-Chinchwad',
        '15': 'Nashik', '16': 'Ahmednagar', '17': 'Shrirampur',
        '18': 'Dhule', '19': 'Jalgaon', '20': 'Aurangabad',
        '21': 'Jalna', '22': 'Parbhani', '23': 'Beed',
        '24': 'Latur', '25': 'Osmanabad', '26': 'Nanded',
        '27': 'Amravati', '28': 'Buldhana', '29': 'Yavatmal',
        '30': 'Akola', '31': 'Nagpur Urban', '32': 'Wardha',
        '33': 'Gadchiroli', '34': 'Chandrapur', '35': 'Gondia',
        '36': 'Bhandara', '37': 'Washim', '38': 'Hingoli',
        '39': 'Nandurbar', '40': 'Nagpur Rural', '41': 'Malegaon',
        '42': 'Baramati', '43': 'Navi Mumbai',
        '44': 'Ambejogai', '45': 'Akluj', '46': 'Panvel',
        '47': 'Borivali', '48': 'Virar', '49': 'Nagpur East',
        '50': 'Karad', '51': 'Mumbai South',
    },
    'DL': {
        '01': 'Mall Road', '02': 'Tilak Marg', '03': 'Sheikh Sarai',
        '04': 'Janakpuri', '05': 'Loni Road', '06': 'Sarai Kale Khan',
        '07': 'Mayur Vihar', '08': 'Wazirpur', '09': 'Dwarka',
        '10': 'Raja Garden', '11': 'Rohini', '12': 'Vasant Vihar',
        '13': 'Surajmal Vihar', '14': 'Sonipat Road', '15': 'Burari',
        '16': 'Shahdara',
    },
    'TN': {
        '01': 'Chennai Central', '02': 'Chennai North-West',
        '03': 'Chennai North-East', '04': 'Chennai South',
        '05': 'Chennai North', '06': 'Chennai South-West',
        '07': 'Chennai South-East', '09': 'Krishnagiri',
        '10': 'Chennai East', '11': 'Chennai West',
        '12': 'Anna Nagar', '13': 'T Nagar', '14': 'Sholinganallur',
        '15': 'Kanchipuram', '16': 'Tambaram', '18': 'Tiruvallur',
        '19': 'Poonamallee', '20': 'Vellore', '21': 'Tiruvannamalai',
        '22': 'Viluppuram', '23': 'Salem West', '24': 'Namakkal',
        '25': 'Dharmapuri', '26': 'Erode', '27': 'Coimbatore North',
        '28': 'Coimbatore South', '29': 'Tiruppur', '30': 'Nilgiris',
        '31': 'Cuddalore', '32': 'Nagapattinam', '33': 'Thanjavur',
        '34': 'Tiruchirappalli', '35': 'Karur', '36': 'Pudukkottai',
        '37': 'Sivagangai', '38': 'Ramanathapuram', '39': 'Virudhunagar',
        '40': 'Madurai Central', '41': 'Madurai South', '43': 'Theni',
        '45': 'Dindigul', '47': 'Tirunelveli', '48': 'Thoothukkudi',
        '49': 'Kanyakumari', '59': 'Salem East',
    },
    'TS': {
        '07': 'Hyderabad Central', '08': 'Hyderabad North',
        '09': 'Hyderabad South', '10': 'Hyderabad East',
        '11': 'Hyderabad West', '12': 'Cyberabad',
        '13': 'Rangareddy', '14': 'Medchal', '28': 'Khammam',
        '29': 'Nalgonda', '30': 'Mahabubnagar', '31': 'Nizamabad',
        '32': 'Karimnagar', '33': 'Adilabad', '34': 'Warangal',
    },
    'AP': {
        '01': 'Adilabad', '05': 'Vijayawada', '07': 'Eluru',
        '09': 'Tirupati', '16': 'Vijayawada East', '37': 'Visakhapatnam',
    },
    'UP': {
        '14': 'Ghaziabad', '15': 'Meerut', '16': 'Noida (Gautam Buddha Nagar)',
        '32': 'Lucknow', '65': 'Varanasi', '70': 'Allahabad/Prayagraj',
        '77': 'Greater Noida', '78': 'Kanpur',
    },
    'GJ': {
        '01': 'Ahmedabad', '02': 'Mehsana', '03': 'Rajkot',
        '04': 'Bhavnagar', '05': 'Surat', '06': 'Vadodara',
        '07': 'Nadiad', '08': 'Palanpur', '09': 'Himmatnagar',
        '18': 'Gandhinagar', '27': 'Ahmedabad East',
    },
    'HR': {
        '01': 'Ambala', '02': 'Jagadhri', '03': 'Panchkula',
        '05': 'Karnal', '06': 'Panipat', '10': 'Sonipat',
        '12': 'Faridabad', '13': 'Palwal', '14': 'Gurugram',
        '26': 'Gurgaon', '36': 'Faridabad NIT', '72': 'Rewari',
    },
    'WB': {
        '01': 'Kolkata South', '02': 'Kolkata North',
        '03': 'Barrackpore', '04': 'Chinsurah', '20': 'Howrah',
        '24': 'Asansol', '26': 'Durgapur', '63': 'Siliguri',
    },
    'KL': {
        '01': 'Thiruvananthapuram', '02': 'Kollam', '03': 'Pathanamthitta',
        '04': 'Alappuzha', '05': 'Kottayam', '06': 'Idukki',
        '07': 'Ernakulam', '08': 'Thrissur', '09': 'Palakkad',
        '10': 'Malappuram', '11': 'Kozhikode', '12': 'Wayanad',
        '13': 'Kannur', '14': 'Kasaragod',
    },
    'RJ': {
        '01': 'Ajmer', '02': 'Alwar', '04': 'Bharatpur',
        '07': 'Churu', '14': 'Jaipur South', '20': 'Kota',
        '27': 'Sikar', '45': 'Jaipur East', '47': 'Jodhpur',
    },
    'MP': {
        '04': 'Bhopal', '07': 'Gwalior', '09': 'Indore',
        '19': 'Jabalpur', '20': 'Ujjain',
    },
    'CH': {
        '01': 'Chandigarh',
    },
    'PY': {
        '01': 'Puducherry',
    },
    'GA': {
        '01': 'Panaji', '02': 'Margao', '03': 'Mapusa',
    },
}


# =====================================================================
# MAIN DECODER FUNCTION
# =====================================================================
def decode_plate(plate_text, vehicle_class_id=None):
    """
    Parse a license plate and return a dict of info.

    Args:
        plate_text: cleaned plate string, e.g. "KA02MN1826"
        vehicle_class_id: optional COCO class ID from YOLO (2=car, 3=bike,
                          5=bus, 7=truck) for vehicle type

    Returns:
        dict with keys: state, rto_code, city, category, series,
                        vehicle_type, valid
    """
    info = {
        'plate': plate_text,
        'state': 'Unknown',
        'rto_code': '',
        'city': 'Unknown',
        'category': 'Private',
        'series': '',
        'vehicle_type': vehicle_class_to_type(vehicle_class_id),
        'valid': False,
    }

    if not plate_text or len(plate_text) < 6:
        return info

    # Special case: Bharat series (BH) — format is YY BH #### LL
    # e.g. "22BH1234AA"
    bh_match = re.match(r'^(\d{2})BH(\d{4})([A-Z]{1,2})$', plate_text)
    if bh_match:
        year, num, letters = bh_match.groups()
        info.update({
            'state': 'Bharat Series (All India)',
            'rto_code': f'BH (20{year})',
            'city': 'Central Government Registration',
            'category': 'BH-Series (Transferable)',
            'series': f'{num} {letters}',
            'valid': True,
        })
        return info

    # Standard Indian format: LL-DD-LL-DDDD (10 chars)
    # or sometimes LL-DD-L-DDDD (9 chars, older)
    std_match = re.match(r'^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$',
                         plate_text)
    if not std_match:
        # Try looser pattern
        std_match = re.match(r'^([A-Z]{2})(\d{1,2})([A-Z]*)(\d+)$',
                             plate_text)

    if std_match:
        state_code, rto_num, series_letters, series_num = std_match.groups()

        # Normalize RTO number to 2 digits
        rto_num_padded = rto_num.zfill(2)

        info['state'] = STATE_NAMES.get(state_code, f'Unknown ({state_code})')
        info['rto_code'] = f'{state_code}-{rto_num_padded}'
        info['series'] = f'{series_letters} {series_num}'.strip()

        # City lookup
        state_rtos = RTO_CITIES.get(state_code, {})
        info['city'] = state_rtos.get(rto_num_padded, 'Unknown RTO')

        # Category: check series letters for commercial indicators
        info['category'] = detect_plate_category(series_letters, plate_text)

        info['valid'] = (state_code in STATE_NAMES)

    return info


def vehicle_class_to_type(class_id):
    """Map COCO class ID from YOLO to readable vehicle type."""
    mapping = {
        2: 'Car',
        3: 'Motorcycle',
        5: 'Bus',
        7: 'Truck',
    }
    if class_id is None:
        return 'Vehicle'
    return mapping.get(class_id, 'Vehicle')


def detect_plate_category(series_letters, full_plate):
    """
    Classify the plate category based on series letters and format.

    Indian plate categories:
    - Private: regular LL-DD-LL-DDDD
    - Commercial: usually starts with T or has Y/Z series
    - Taxi/Yellow board: Y series
    - Government: series starting with G
    - BH: Bharat Series (handled separately)
    """
    if not series_letters:
        return 'Private'

    first = series_letters[0] if series_letters else ''

    # These are heuristics — not 100% but typically right
    if first == 'Y':
        return 'Commercial (Taxi/Yellow Board)'
    elif first == 'T':
        return 'Commercial (Transport)'
    elif first == 'G':
        return 'Government'
    elif first in ('Z',):
        return 'Commercial'
    else:
        return 'Private'


# =====================================================================
# SELF-TEST
# =====================================================================
if __name__ == "__main__":
    test_plates = [
        ('KA02MN1826', 2),       # Karnataka car, Bangalore-Rajajinagar
        ('MH12AB1234', 2),       # Maharashtra car, Pune
        ('DL08CAF5030', 2),      # Delhi commercial
        ('TN07BH7890', 2),       # Tamil Nadu
        ('22BH1234AA', 2),       # Bharat series
        ('UP16AB1234', 3),       # UP motorcycle, Noida
        ('KA03Y1234', 2),        # Karnataka taxi
    ]

    print("Plate Info Decoder — Self Test\n" + "=" * 50)
    for plate, cls in test_plates:
        info = decode_plate(plate, cls)
        print(f"\nPlate: {plate}")
        print(f"  State:    {info['state']}")
        print(f"  RTO:      {info['rto_code']}  ({info['city']})")
        print(f"  Category: {info['category']}")
        print(f"  Series:   {info['series']}")
        print(f"  Type:     {info['vehicle_type']}")
