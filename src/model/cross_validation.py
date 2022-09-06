#################################################################
#
# Cross-Validation Schema
#
#   ┌─────────────┐                       ┌───────┐
#   │  Parameters │               ┌───────┤Dataset├──────┐
#   └──────┬──────┘               │       └───────┘      │
#          │                      │                      │
#          ▼                      ▼                      ▼
# ┌─────────────────┐      ┌─────────────┐          ┌─────────┐
# │Cross-Validation │◄─────┤Training data│          │Test data│
# └────────┬────────┘      └──────┬──────┘          └────┬────┘
#          │                      │                      │
#          ▼                      ▼                      │
#  ┌───────────────┐      ┌───────────────┐              │
#  │Best parameters├─────►│Retrained model│              │
#  └───────────────┘      └───────┬───────┘              │
#                                 │                      │
#                                 │                      │
#                                 │  ┌────────────────┐  │
#                                 └─►│Final evaluation│◄─┘
#                                    └────────────────┘
#
###############################################################
