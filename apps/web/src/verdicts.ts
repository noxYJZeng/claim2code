import type { VerificationStatus } from "./api";

// How each verdict is written: a margin mark in its own ink colour,
// as a reviewer would annotate a proof.

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

export const STATUS_MARK: Record<VerificationStatus, string> = {
  VERIFIED: "✓",
  PARTIALLY_VERIFIED: "◐",
  CONTRADICTED: "✕",
  UNVERIFIED: "?",
};

export const STATUS_INK: Record<VerificationStatus, string> = {
  VERIFIED: "text-verdict-ok",
  PARTIALLY_VERIFIED: "text-verdict-partial",
  CONTRADICTED: "text-verdict-bad",
  UNVERIFIED: "text-verdict-none",
};

export const STATUS_EDGE: Record<VerificationStatus, string> = {
  VERIFIED: "border-verdict-ok",
  PARTIALLY_VERIFIED: "border-verdict-partial",
  CONTRADICTED: "border-verdict-bad",
  UNVERIFIED: "border-verdict-none",
};
