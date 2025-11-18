"""
Agricultural Disease Data Structures
=====================================

"Bad programmers worry about the code. Good programmers worry about data structures."

This module defines the core data structures for the agricultural disease recognition system.
The key insight: 61 categories are NOT flat - they have a natural hierarchy:
    Crop Type (10) -> Disease (28, including "None" for healthy) -> Severity (0/1/2)

Good taste: eliminate special cases by using a unified representation.
"""

import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class SeverityLevel(IntEnum):
    """Disease severity levels (ordered)"""

    HEALTHY = 0  # No disease
    GENERAL = 1  # Mild/General disease
    SERIOUS = 2  # Severe disease


@dataclass(frozen=True)
class DiseaseLabel:
    """
    Unified three-tuple representation for all labels.

    This eliminates "Healthy" as a special case:
    - Apple Healthy        -> DiseaseLabel(crop="Apple", disease=None, severity=0)
    - Apple Scab General   -> DiseaseLabel(crop="Apple", disease="Scab", severity=1)
    - Apple Scab Serious   -> DiseaseLabel(crop="Apple", disease="Scab", severity=2)

    Good taste: one representation for all cases, no if/else branches.
    """

    crop_type: str  # One of 10 crops
    disease: Optional[str]  # None for healthy, disease name otherwise
    severity: SeverityLevel  # 0=healthy, 1=general, 2=serious
    label_61: int  # Original 61-class label ID

    @property
    def is_healthy(self) -> bool:
        """Check if this is a healthy sample"""
        return self.disease is None

    @property
    def has_general_level(self) -> bool:
        """Check if disease has mild/general severity"""
        return self.severity == SeverityLevel.GENERAL

    @property
    def has_serious_level(self) -> bool:
        """Check if disease has serious severity"""
        return self.severity == SeverityLevel.SERIOUS

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "crop_type": self.crop_type,
            "disease": self.disease,
            "severity": int(self.severity),
            "label_61": self.label_61,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DiseaseLabel":
        """Create from dictionary"""
        return cls(
            crop_type=data["crop_type"],
            disease=data.get("disease"),
            severity=SeverityLevel(data["severity"]),
            label_61=data["label_61"],
        )


# Label mappings: the source of truth
# This eliminates hardcoded magic numbers throughout the codebase

LABEL_61_TO_NAME = {
    0: "Apple Healthy",
    1: "Apple Scab (General)",
    2: "Apple Scab (Serious)",
    3: "Apple Frogeye Spot",
    4: "Cedar Apple Rust (General)",
    5: "Cedar Apple Rust (Serious)",
    6: "Cherry Healthy",
    7: "Cherry Powdery Mildew (General)",
    8: "Cherry Powdery Mildew (Serious)",
    9: "Corn Healthy",
    10: "Cercospora Zeaemaydis Tehon and Daniels (General)",
    11: "Cercospora Zeaemaydis Tehon and Daniels (Serious)",
    12: "Corn Puccinia Polysora (General)",
    13: "Corn Puccinia Polysora (Serious)",
    14: "Corn Curvularia Leaf Spot (Fungus, General)",
    15: "Corn Curvularia Leaf Spot (Fungus, Serious)",
    16: "Maize Dwarf Mosaic Virus",
    17: "Grape Healthy",
    18: "Grape Black Rot (Fungus, General)",
    19: "Grape Black Rot (Fungus, Serious)",
    20: "Grape Black Measles (Fungus, General)",
    21: "Grape Black Measles (Fungus, Serious)",
    22: "Grape Leaf Blight (Fungus, General)",
    23: "Grape Leaf Blight (Fungus, Serious)",
    24: "Citrus Healthy",
    25: "Citrus Greening (General)",
    26: "Citrus Greening (Serious)",
    27: "Peach Healthy",
    28: "Peach Bacterial Spot (General)",
    29: "Peach Bacterial Spot (Serious)",
    30: "Pepper Healthy",
    31: "Pepper Scab (General)",
    32: "Pepper Scab (Serious)",
    33: "Potato Healthy",
    34: "Potato Early Blight (Fungus, General)",
    35: "Potato Early Blight (Fungus, Serious)",
    36: "Potato Late Blight (Fungus, General)",
    37: "Potato Late Blight (Fungus, Serious)",
    38: "Strawberry Healthy",
    39: "Strawberry Scorch (General)",
    40: "Strawberry Scorch (Serious)",
    41: "Tomato Healthy",
    42: "Tomato Powdery Mildew (General)",
    43: "Tomato Powdery Mildew (Serious)",
    44: "Tomato Bacterial Spot (Bacteria, General)",
    45: "Tomato Bacterial Spot (Bacteria, Serious)",
    46: "Tomato Early Blight (Fungus, General)",
    47: "Tomato Early Blight (Fungus, Serious)",
    48: "Tomato Late Blight (Water Mold, General)",
    49: "Tomato Late Blight (Water Mold, Serious)",
    50: "Tomato Leaf Mold (Fungus, General)",
    51: "Tomato Leaf Mold (Fungus, Serious)",
    52: "Tomato Target Spot (Bacteria, General)",
    53: "Tomato Target Spot (Bacteria, Serious)",
    54: "Tomato Septoria Leaf Spot (Fungus, General)",
    55: "Tomato Septoria Leaf Spot (Fungus, Serious)",
    56: "Tomato Spider Mite Damage (General)",
    57: "Tomato Spider Mite Damage (Serious)",
    58: "Tomato Yellow Leaf Curl Virus (General)",
    59: "Tomato Yellow Leaf Curl Virus (Serious)",
    60: "Tomato Mosaic Virus",
}

# Crop types (10 classes)
CROP_TYPES = [
    "Apple",
    "Cherry",
    "Corn",
    "Grape",
    "Citrus",
    "Peach",
    "Pepper",
    "Potato",
    "Strawberry",
    "Tomato",
]

CROP_TO_ID = {crop: idx for idx, crop in enumerate(CROP_TYPES)}
ID_TO_CROP = {idx: crop for crop, idx in CROP_TO_ID.items()}

# Disease types (28 classes: 27 diseases + 1 for "None"/healthy)
DISEASE_TYPES = [
    "None",  # Healthy (index 0)
    "Scab",
    "Frogeye Spot",
    "Cedar Apple Rust",
    "Powdery Mildew",
    "Cercospora Zeaemaydis Tehon and Daniels",
    "Puccinia Polysora",
    "Curvularia Leaf Spot",
    "Dwarf Mosaic Virus",
    "Black Rot",
    "Black Measles",
    "Leaf Blight",
    "Greening",
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Scorch",
    "Leaf Mold",
    "Target Spot",
    "Septoria Leaf Spot",
    "Spider Mite Damage",
    "Yellow Leaf Curl Virus",
    "Mosaic Virus",
]

DISEASE_TO_ID = {disease: idx for idx, disease in enumerate(DISEASE_TYPES)}
ID_TO_DISEASE = {idx: disease for disease, idx in DISEASE_TO_ID.items()}

# Special cases: labels that don't start with crop name
SPECIAL_LABEL_MAPPINGS = {
    "Cedar Apple Rust (General)": ("Apple", "Cedar Apple Rust", SeverityLevel.GENERAL, 4),
    "Cedar Apple Rust (Serious)": ("Apple", "Cedar Apple Rust", SeverityLevel.SERIOUS, 5),
    "Cercospora Zeaemaydis Tehon and Daniels (General)": (
        "Corn",
        "Cercospora Zeaemaydis Tehon and Daniels",
        SeverityLevel.GENERAL,
        10,
    ),
    "Cercospora Zeaemaydis Tehon and Daniels (Serious)": (
        "Corn",
        "Cercospora Zeaemaydis Tehon and Daniels",
        SeverityLevel.SERIOUS,
        11,
    ),
    "Maize Dwarf Mosaic Virus": ("Corn", "Dwarf Mosaic Virus", SeverityLevel.GENERAL, 16),
}


def parse_label_name(label_name: str) -> DiseaseLabel:
    """
    Parse a label name into structured DiseaseLabel.

    Examples:
        "Apple Healthy" -> DiseaseLabel(crop="Apple", disease=None, severity=0, ...)
        "Apple Scab (General)" -> DiseaseLabel(crop="Apple", disease="Scab", severity=1, ...)
        "Apple Frogeye Spot" -> DiseaseLabel(crop="Apple", disease="Frogeye Spot", severity=1, ...)

    Good taste: handle all cases uniformly, no special branches.
    """
    # Check special cases first (labels not starting with crop name)
    if label_name in SPECIAL_LABEL_MAPPINGS:
        crop_type, disease, severity, label_id = SPECIAL_LABEL_MAPPINGS[label_name]
        return DiseaseLabel(
            crop_type=crop_type, disease=disease, severity=severity, label_61=label_id
        )

    # Find which crop it belongs to
    # Sort by length descending to match longest first (e.g., "Strawberry" before "Straw")
    crop_type = None
    for crop in sorted(CROP_TYPES, key=len, reverse=True):
        if label_name.startswith(crop):
            crop_type = crop
            break

    if crop_type is None:
        raise ValueError(f"Unknown crop in label: {label_name}")

    # Remove crop prefix
    remainder = label_name[len(crop_type) :].strip()

    # Check if healthy
    if remainder == "Healthy":
        disease = None
        severity = SeverityLevel.HEALTHY
    else:
        # Check severity level
        if remainder.endswith("(General)"):
            severity = SeverityLevel.GENERAL
            disease = remainder.replace("(General)", "").strip()
        elif remainder.endswith("(Serious)"):
            severity = SeverityLevel.SERIOUS
            disease = remainder.replace("(Serious)", "").strip()
        else:
            # Single-level disease (no General/Serious distinction)
            severity = SeverityLevel.GENERAL
            disease = remainder.strip()

        # Clean up disease name (remove pathogen type if present)
        if "(Fungus" in disease or "(Bacteria" in disease or "(Water Mold" in disease:
            disease = disease.split("(")[0].strip()

    # Find label_61 ID
    label_61 = None
    for lid, lname in LABEL_61_TO_NAME.items():
        if lname == label_name:
            label_61 = lid
            break

    if label_61 is None:
        raise ValueError(f"Label name not found in mapping: {label_name}")

    return DiseaseLabel(crop_type=crop_type, disease=disease, severity=severity, label_61=label_61)


def build_label_hierarchy() -> Dict[int, DiseaseLabel]:
    """
    Build the complete label hierarchy mapping.

    Returns:
        Dictionary mapping label_61 ID -> DiseaseLabel structure

    This is the single source of truth for label conversions.
    Build once, use everywhere - no repeated parsing.
    """
    hierarchy = {}

    for label_id, label_name in LABEL_61_TO_NAME.items():
        disease_label = parse_label_name(label_name)
        hierarchy[label_id] = disease_label

    return hierarchy


def get_severity_4class_mapping() -> Dict[int, int]:
    """
    Map 61-class labels to 4-class severity labels (Task 3).

    4 classes:
        0: Healthy
        1: Mild (General level diseases)
        2: Moderate (could be mapped from General for some diseases)
        3: Severe (Serious level diseases)

    For simplicity, we use:
        0: Healthy
        1: General level
        2: General level (duplicate, but keeps 4-class structure)
        3: Serious level

    A better approach might be to use actual disease progression data,
    but we don't have that information in the dataset.
    """
    hierarchy = build_label_hierarchy()
    mapping = {}

    for label_id, disease_label in hierarchy.items():
        if disease_label.severity == SeverityLevel.HEALTHY:
            mapping[label_id] = 0
        elif disease_label.severity == SeverityLevel.GENERAL:
            mapping[label_id] = 1
        else:  # SERIOUS
            mapping[label_id] = 3

    return mapping


# Build the hierarchy once at module load time
LABEL_HIERARCHY = build_label_hierarchy()


def get_multitask_labels(label_61: int) -> Dict[str, int]:
    """
    Convert 61-class label to multi-task labels.

    Returns:
        Dictionary with keys: 'label_61', 'crop', 'disease', 'severity'

    This is used for multi-task learning (Task 4).
    """
    disease_label = LABEL_HIERARCHY[label_61]

    return {
        "label_61": label_61,
        "crop": CROP_TO_ID[disease_label.crop_type],
        "disease": DISEASE_TO_ID.get(disease_label.disease, 0),  # 0 for None/healthy
        "severity": int(disease_label.severity),
    }


def print_label_statistics():
    """Print statistics about the label hierarchy (for debugging)"""
    print("Label Hierarchy Statistics")
    print("=" * 50)
    print(f"Total labels: {len(LABEL_HIERARCHY)}")
    print(f"Crop types: {len(CROP_TYPES)}")
    print(f"Disease types: {len(DISEASE_TYPES)}")
    print()

    # Count by crop
    crop_counts = {}
    for label in LABEL_HIERARCHY.values():
        crop_counts[label.crop_type] = crop_counts.get(label.crop_type, 0) + 1

    print("Labels per crop:")
    for crop, count in sorted(crop_counts.items()):
        print(f"  {crop:12s}: {count:2d}")
    print()

    # Count by severity
    severity_counts = {0: 0, 1: 0, 2: 0}
    for label in LABEL_HIERARCHY.values():
        severity_counts[int(label.severity)] += 1

    print("Labels by severity:")
    print(f"  Healthy:  {severity_counts[0]:2d}")
    print(f"  General:  {severity_counts[1]:2d}")
    print(f"  Serious:  {severity_counts[2]:2d}")


if __name__ == "__main__":
    # Test the data structures
    print("Testing Agricultural Disease Data Structures")
    print("=" * 50)
    print()

    # Test parsing
    test_labels = [
        "Apple Healthy",
        "Apple Scab (General)",
        "Apple Scab (Serious)",
        "Tomato Mosaic Virus",
    ]

    for label_name in test_labels:
        label_id = None
        for lid, lname in LABEL_61_TO_NAME.items():
            if lname == label_name:
                label_id = lid
                break

        disease_label = parse_label_name(label_name)
        print(f"Label {label_id}: {label_name}")
        print(f"  Crop: {disease_label.crop_type}")
        print(f"  Disease: {disease_label.disease}")
        print(f"  Severity: {disease_label.severity.name}")
        print()

    # Print statistics
    print_label_statistics()

    # Test multi-task conversion
    print()
    print("Multi-task label conversion example:")
    print("-" * 50)
    label_61 = 2  # Apple Scab (Serious)
    mt_labels = get_multitask_labels(label_61)
    print(f"Label 61 ID: {label_61} ({LABEL_61_TO_NAME[label_61]})")
    print(f"Multi-task labels: {mt_labels}")
    print()
    print("Good taste: one unified representation, no special cases.")
