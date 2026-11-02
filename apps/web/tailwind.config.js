/** @type {import('tailwindcss').Config} */
// Framed-sheet app style: a stone-tinted frame (the paper) holds the
// sidebar and top bar; the work sits on a white sheet. One teal hue
// marks the trace between a claim's paper passage and its code, and the
// selected claim; verdict colours are status and stay on data.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        frame: "#f3f3f0",
        sheet: "#ffffff",
        raised: "#f6f6f4",
        active: "#e9e9e5",
        line: { DEFAULT: "#e6e6e2", rule: "#f0f0ed", control: "#cdcdc8" },
        ink: { DEFAULT: "#131416", muted: "#5d5f60", faint: "#6b6b6b" },
        accent: { DEFAULT: "#1b6c80", wash: "#e7f0f2" },
        positive: { DEFAULT: "#1e7b4f", bg: "#dcfae5", ink: "#04250e" },
        negative: { DEFAULT: "#c4242a", bg: "#ffedea", ink: "#831c1e" },
        warning: { DEFAULT: "#a8650c", bg: "#fffbeb", ink: "#92400e", line: "#fde68a" },
        neutral: { bg: "#ebedee", ink: "#353736" },
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', "Inter", "system-ui", "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "Menlo", "monospace"],
      },
      borderRadius: { card: "8px", control: "6px", chip: "4px", sheet: "16px" },
    },
  },
  plugins: [],
};
