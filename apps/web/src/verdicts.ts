import type { VerificationStatus } from "./api";

// Verdicts are status: a pale pill with dark text of the same hue in
// lists, and a solid cell in the claim strip. Nothing else on the
// screen uses these colours.

export const STATUSES: VerificationStatus[] = [
  "VERIFIED",
  "PARTIALLY_VERIFIED",
  "CONTRADICTED",
  "UNVERIFIED",
];

export const STATUS_LABEL: Record<VerificationStatus, string> = {
  VERIFIED: "Verified",
  PARTIALLY_VERIFIED: "Partially verified",
  CONTRADICTED: "Contradicted",
  UNVERIFIED: "Unverified",
};

export const STATUS_PILL: Record<VerificationStatus, string> = {
  VERIFIED: "bg-positive-bg text-positive-ink",
  PARTIALLY_VERIFIED: "bg-warning-bg text-warning-ink",
  CONTRADICTED: "bg-negative-bg text-negative-ink",
  UNVERIFIED: "bg-neutral-bg text-neutral-ink",
};

export const STATUS_CELL: Record<VerificationStatus, string> = {
  VERIFIED: "bg-positive",
  PARTIALLY_VERIFIED: "bg-warning",
  CONTRADICTED: "bg-negative",
  UNVERIFIED: "bg-ink-faint",
};
