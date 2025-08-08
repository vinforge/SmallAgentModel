"""
Token Roles for Table Processing
================================

Defines the 9 semantic token roles for table understanding based on the
TableMoE paper and enhanced for SAM's specific needs.

These roles enable SAM to understand the semantic meaning of each cell
in a table, going beyond simple text extraction to true structural comprehension.
"""

from enum import Enum
from typing import Dict, List, Optional, Set
from dataclasses import dataclass


class TokenRole(Enum):
    """
    Semantic roles for table tokens based on TableMoE architecture.
    
    These 9 roles provide comprehensive coverage of table semantics:
    """
    
    # Core structural roles
    HEADER = "HEADER"           # Column/row headers, titles
    DATA = "DATA"               # Regular data cells
    EMPTY = "EMPTY"             # Empty cells, null values
    
    # Analytical roles
    TOTAL = "TOTAL"             # Sum, subtotal, grand total cells
    FORMULA = "FORMULA"         # Calculated values, formulas
    AXIS = "AXIS"               # Row/column labels, indices
    
    # Contextual roles
    CAPTION = "CAPTION"         # Table captions, descriptions
    METADATA = "METADATA"       # Source info, notes, footnotes
    OTHER = "OTHER"             # Unclassified or special content


@dataclass
class RoleDefinition:
    """Detailed definition of a semantic role."""
    role: TokenRole
    description: str
    examples: List[str]
    patterns: List[str]
    context_indicators: List[str]
    confidence_factors: Dict[str, float]


# Comprehensive role definitions
SEMANTIC_ROLES: Dict[TokenRole, RoleDefinition] = {
    
    TokenRole.HEADER: RoleDefinition(
        role=TokenRole.HEADER,
        description="Column headers, row headers, and table titles that describe data categories",
        examples=["Name", "Date", "Amount", "Q1 2023", "Product Category", "Total Sales"],
        patterns=["^[A-Z][a-z]+", ".*[Hh]eader.*", ".*[Tt]itle.*", ".*[Cc]olumn.*"],
        context_indicators=["first_row", "first_column", "bold_text", "centered_text"],
        confidence_factors={"position_weight": 0.4, "formatting_weight": 0.3, "content_weight": 0.3}
    ),
    
    TokenRole.DATA: RoleDefinition(
        role=TokenRole.DATA,
        description="Regular data cells containing the primary information",
        examples=["John Smith", "2023-01-15", "1,250.00", "Active", "Product A"],
        patterns=["\\d+", "\\d+\\.\\d+", "[A-Za-z0-9\\s]+", "\\$\\d+"],
        context_indicators=["body_cell", "regular_formatting", "data_pattern"],
        confidence_factors={"position_weight": 0.2, "formatting_weight": 0.2, "content_weight": 0.6}
    ),
    
    TokenRole.EMPTY: RoleDefinition(
        role=TokenRole.EMPTY,
        description="Empty cells, null values, or placeholder content",
        examples=["", "N/A", "NULL", "-", "TBD", "..."],
        patterns=["^$", "^\\s*$", "^N/?A$", "^NULL$", "^-+$", "^\\.+$"],
        context_indicators=["empty_content", "placeholder_text", "null_indicator"],
        confidence_factors={"position_weight": 0.1, "formatting_weight": 0.1, "content_weight": 0.8}
    ),
    
    TokenRole.TOTAL: RoleDefinition(
        role=TokenRole.TOTAL,
        description="Sum, subtotal, grand total, and aggregated values",
        examples=["Total: $5,000", "Subtotal", "Grand Total", "Sum", "Average: 85%"],
        patterns=[".*[Tt]otal.*", ".*[Ss]um.*", ".*[Aa]verage.*", ".*[Ss]ubtotal.*"],
        context_indicators=["bottom_row", "summary_section", "bold_text", "calculation_result"],
        confidence_factors={"position_weight": 0.4, "formatting_weight": 0.2, "content_weight": 0.4}
    ),
    
    TokenRole.FORMULA: RoleDefinition(
        role=TokenRole.FORMULA,
        description="Calculated values, formulas, and computed results",
        examples=["=SUM(A1:A10)", "=B2*C2", "Calculated", "Formula Result", "Auto-calculated"],
        patterns=["^=.*", ".*[Cc]alculated.*", ".*[Ff]ormula.*", ".*[Cc]omputed.*"],
        context_indicators=["formula_cell", "calculated_value", "computation_indicator"],
        confidence_factors={"position_weight": 0.2, "formatting_weight": 0.2, "content_weight": 0.6}
    ),
    
    TokenRole.AXIS: RoleDefinition(
        role=TokenRole.AXIS,
        description="Row and column labels, indices, and dimensional indicators",
        examples=["Row 1", "Column A", "Index", "ID", "Seq", "Item #"],
        patterns=["[Rr]ow\\s+\\d+", "[Cc]ol\\s+[A-Z]", ".*[Ii]ndex.*", ".*ID.*", "#\\d+"],
        context_indicators=["index_column", "sequence_indicator", "identifier_cell"],
        confidence_factors={"position_weight": 0.5, "formatting_weight": 0.2, "content_weight": 0.3}
    ),
    
    TokenRole.CAPTION: RoleDefinition(
        role=TokenRole.CAPTION,
        description="Table captions, titles, and descriptive text",
        examples=["Table 1: Sales Report", "Figure 2.1", "Monthly Revenue Summary", "Data Source: Internal"],
        patterns=["[Tt]able\\s+\\d+", "[Ff]igure\\s+\\d+", ".*[Ss]ummary.*", ".*[Rr]eport.*"],
        context_indicators=["above_table", "below_table", "caption_formatting", "title_text"],
        confidence_factors={"position_weight": 0.4, "formatting_weight": 0.3, "content_weight": 0.3}
    ),
    
    TokenRole.METADATA: RoleDefinition(
        role=TokenRole.METADATA,
        description="Source information, notes, footnotes, and auxiliary data",
        examples=["*Note: Data as of Dec 2023", "Source: Company Database", "Updated: 2023-12-01"],
        patterns=["\\*.*", ".*[Ss]ource.*", ".*[Nn]ote.*", ".*[Uu]pdated.*", ".*[Ff]ootnote.*"],
        context_indicators=["footnote_area", "metadata_section", "source_info", "annotation"],
        confidence_factors={"position_weight": 0.3, "formatting_weight": 0.3, "content_weight": 0.4}
    ),
    
    TokenRole.OTHER: RoleDefinition(
        role=TokenRole.OTHER,
        description="Unclassified content or special table elements",
        examples=["Special formatting", "Merged cell content", "Custom elements"],
        patterns=[".*"],  # Catch-all pattern
        context_indicators=["unclassified", "special_formatting", "merged_cell"],
        confidence_factors={"position_weight": 0.2, "formatting_weight": 0.2, "content_weight": 0.6}
    )
}


def get_role_by_name(role_name: str) -> Optional[TokenRole]:
    """Get TokenRole enum by string name."""
    try:
        return TokenRole(role_name.upper())
    except ValueError:
        return None


def get_all_roles() -> List[TokenRole]:
    """Get all available token roles."""
    return list(TokenRole)


def get_role_definition(role: TokenRole) -> RoleDefinition:
    """Get detailed definition for a specific role."""
    return SEMANTIC_ROLES[role]


def get_role_hierarchy() -> Dict[str, List[TokenRole]]:
    """Get hierarchical grouping of roles for analysis."""
    return {
        "structural": [TokenRole.HEADER, TokenRole.DATA, TokenRole.EMPTY],
        "analytical": [TokenRole.TOTAL, TokenRole.FORMULA, TokenRole.AXIS],
        "contextual": [TokenRole.CAPTION, TokenRole.METADATA, TokenRole.OTHER]
    }


def is_content_role(role: TokenRole) -> bool:
    """Check if role represents actual content (not empty or metadata)."""
    content_roles = {TokenRole.HEADER, TokenRole.DATA, TokenRole.TOTAL, 
                    TokenRole.FORMULA, TokenRole.AXIS, TokenRole.CAPTION}
    return role in content_roles


def get_role_priority(role: TokenRole) -> int:
    """Get priority score for role (higher = more important for analysis)."""
    priority_map = {
        TokenRole.HEADER: 9,
        TokenRole.DATA: 8,
        TokenRole.TOTAL: 7,
        TokenRole.FORMULA: 6,
        TokenRole.AXIS: 5,
        TokenRole.CAPTION: 4,
        TokenRole.METADATA: 3,
        TokenRole.EMPTY: 2,
        TokenRole.OTHER: 1
    }
    return priority_map.get(role, 0)
