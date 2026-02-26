import { Routes, Route } from "react-router-dom";
import { ThemeProvider } from "@/components/providers/ThemeProvider";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { Background } from "@/components/layout/Background";
import HomePage from "@/pages/Home";
import DashboardPage from "@/pages/Dashboard";
import AboutPage from "@/pages/About";

function App() {
  return (
    <ThemeProvider>
      <div className="font-sans antialiased min-h-screen flex flex-col">
        <Background />
        <Navbar />
        <main className="flex-1 pt-24">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </ThemeProvider>
  );
}

export default App;
