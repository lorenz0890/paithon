class PaithonError(Exception):
    pass


class ProviderConfigurationError(PaithonError):
    pass


class UnsupportedFunctionError(PaithonError):
    pass


class CodeGenerationError(PaithonError):
    pass


class CodeRepairError(PaithonError):
    pass


class StateMutationError(PaithonError):
    pass


class StateRollbackError(PaithonError):
    pass


class SafetyViolationError(PaithonError):
    pass


class ReviewPromotionError(PaithonError):
    pass


class RuntimePolicyError(PaithonError):
    pass
