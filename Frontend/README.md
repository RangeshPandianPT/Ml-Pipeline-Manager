# MLOps Pipeline Dashboard

This is the frontend dashboard for the ML Pipeline Manager. It provides a visual interface for managing the complete Machine Learning lifecycle, from data ingestion to model deployment and drift monitoring.

## 🚀 Technologies Used

- **React:** UI library
- **TypeScript:** Type-safe JavaScript
- **Vite:** Fast frontend build tool
- **Tailwind CSS:** Utility-first CSS framework
- **Shadcn UI / Radix primitives:** Accessible UI components

## 🛠️ Getting Started

### Prerequisites

Ensure you have Node.js and `npm` (or `bun`/`yarn`) installed.

### Installation

Navigate to the frontend directory and install dependencies:

```bash
cd Frontend
npm install
```

### Running Locally

Start the Vite development server:

```bash
npm run dev
```

The application will typically be available at \`http://localhost:8080\` (or the port specified in your terminal).

## 📂 Project Structure

- `src/pages`: Main application views (e.g., Data Ingestion, Model Training, Drift Monitoring).
- `src/components`: Reusable UI components.
- `src/hooks`: Custom React hooks, including API connection utilities.
- `src/lib`: Utility functions and helpers.

## 🔗 Backend Integration

The frontend is designed to interface with the FastAPI backend of the ML Pipeline Manager. Ensure the backend server is running for full functionality.
