/** @type {import('tailwindcss').Config} */
// A proof-sheet palette: warm paper, ink, and coloured inks for the
// four verdicts. Cited text is marked with a highlighter colour, the
// same in the paper excerpt and in the code.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#f4f1ea",
        surface: "#fbfaf7",
        ink: { DEFAULT: "#1c1a16", 2: "#4f4a42", 3: "#8a8378" },
        rule: { DEFAULT: "#e2dccf", strong: "#c8bfad" },
        accent: "#2c4a86",
        verdict: {
          ok: "#2f6a4a",
          partial: "#9a6512",
          bad: "#a5352a",
          none: "#6f695e",
        },
        hl: "#f2e4a2",
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', "system-ui", "sans-serif"],
        serif: ['"IBM Plex Serif"', "Georgia", "serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "Menlo", "monospace"],
      },
      borderRadius: { sm: "3px" },
    },
  },
  plugins: [],
};
