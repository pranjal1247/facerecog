/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cyber: {
          cyan: "#00f0ff",
          green: "#00ff99",
          amber: "#ff9900",
        }
      }
    },
  },
  plugins: [],
}