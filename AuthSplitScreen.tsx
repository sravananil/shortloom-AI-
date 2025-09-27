import React, { useState } from "react";
import { motion } from "framer-motion";
import myPhoto from "../assets/myphoto.jpg";
import { login, register, storeTokens } from "../auth";

const profilePic = myPhoto;

const GlassCard: React.FC = () => (
  <motion.div
    drag
    dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
    className="relative flex flex-col items-center select-none z-10"
  >
    <div className="w-1 h-12 bg-gradient-to-b from-gray-400 to-transparent rounded-full mb-[-24px]"></div>
    <div className="backdrop-blur-lg bg-white/30 border border-white/40 shadow-2xl rounded-2xl p-6 w-72 flex flex-col items-center transition-all duration-300">
      <img
        src={profilePic}
        alt="Profile"
        className="w-20 h-20 rounded-full border-4 border-blue-400 shadow mb-3 object-cover"
      />
      <div className="text-lg font-bold text-gray-800">ShortLoom</div>
      <div className="text-sm text-gray-600 mt-1 text-center">
        Transform your videos into viral shorts
      </div>
    </div>
  </motion.div>
);

interface AuthSplitScreenProps {
  onLogin?: (token: string) => void;
  onClose?: () => void;
}

const AuthSplitScreen: React.FC<AuthSplitScreenProps> = ({ onLogin, onClose }) => {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
    confirm: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [show, setShow] = useState(true);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setError(null);
    setSuccess(null);
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);
    try {
      const res = await login(form.email, form.password);
      setSuccess("Login successful!");
      if (res.access_token && res.refresh_token) {
        storeTokens({ access_token: res.access_token, refresh_token: res.refresh_token });
        if (onLogin) onLogin(res.access_token);
        setShow(false);
      }
    } catch (err: any) {
      setError(err.response?.data?.message || "Login failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);
    if (form.password !== form.confirm) {
      setError("Passwords do not match.");
      setLoading(false);
      return;
    }
    try {
      const res = await register(form.email, form.password);
      setSuccess("Registration successful! Please login.");
      setMode("login");
      setForm({ ...form, password: "", confirm: "" });
    } catch (err: any) {
      setError(err.response?.data?.message || "Registration failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setShow(false);
    if (onClose) onClose();
  };

  if (!show) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm">
      <div className="relative w-full max-w-sm mx-auto bg-white/10 backdrop-blur-xl rounded-2xl shadow-2xl p-4 md:p-8 glass-card border border-white/20">
        {/* Exit/Close Button */}
        <button
          className="absolute top-2 right-2 w-7 h-7 flex items-center justify-center rounded-lg bg-black/30 text-gray-200 hover:bg-black/50 hover:text-white text-lg font-bold focus:outline-none transition-all duration-150 shadow"
          style={{ lineHeight: 1 }}
          onClick={handleClose}
          aria-label="Close login/register"
        >
          &times;
        </button>
        <h1 className="text-2xl font-bold text-gray-800 mb-6 text-center tracking-tight">
          {mode === "login" ? "Sign in to ShortLoom" : "Create your account"}
        </h1>
        {error && (
          <div className="mb-4 p-3 bg-red-100 border border-red-300 text-red-700 rounded text-center text-sm animate-pulse">{error}</div>
        )}
        {success && (
          <div className="mb-4 p-3 bg-green-100 border border-green-300 text-green-700 rounded text-center text-sm animate-pulse">{success}</div>
        )}
        {mode === "login" ? (
          <form onSubmit={handleLogin} className="space-y-3 md:space-y-4">
            <input
              type="email"
              name="email"
              placeholder="Email"
              className="w-full p-2.5 md:p-3 rounded-lg border border-gray-300 bg-white/60 text-sm md:text-base text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-400 transition"
              value={form.email}
              onChange={handleChange}
              required
              autoComplete="username"
            />
            <input
              type="password"
              name="password"
              placeholder="Password"
              className="w-full p-2.5 md:p-3 rounded-lg border border-gray-300 bg-white/60 text-sm md:text-base text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-400 transition"
              value={form.password}
              onChange={handleChange}
              required
              autoComplete="current-password"
            />
            <button
              type="submit"
              className="w-full bg-gradient-to-r from-blue-500 to-purple-500 text-white py-2.5 md:py-3 rounded-lg font-semibold shadow hover:scale-105 transition disabled:opacity-60 text-sm md:text-base"
              disabled={loading}
            >
              {loading ? "Logging in..." : "Login"}
            </button>
            <div className="text-center mt-3 md:mt-4">
              <button
                type="button"
                className="text-blue-600 hover:underline text-xs md:text-sm"
                onClick={() => setMode("register")}
                disabled={loading}
              >
                Create Account
              </button>
            </div>
          </form>
        ) : (
          <form onSubmit={handleRegister} className="space-y-3 md:space-y-4">
            <input
              type="text"
              name="name"
              placeholder="Name"
              className="w-full p-2.5 md:p-3 rounded-lg border border-gray-300 bg-white/60 text-sm md:text-base text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-400 transition"
              value={form.name}
              onChange={handleChange}
              required
              autoComplete="name"
            />
            <input
              type="email"
              name="email"
              placeholder="Email"
              className="w-full p-2.5 md:p-3 rounded-lg border border-gray-300 bg-white/60 text-sm md:text-base text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-400 transition"
              value={form.email}
              onChange={handleChange}
              required
              autoComplete="username"
            />
            <input
              type="password"
              name="password"
              placeholder="Password"
              className="w-full p-2.5 md:p-3 rounded-lg border border-gray-300 bg-white/60 text-sm md:text-base text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-400 transition"
              value={form.password}
              onChange={handleChange}
              required
              autoComplete="new-password"
            />
            <input
              type="password"
              name="confirm"
              placeholder="Confirm Password"
              className="w-full p-2.5 md:p-3 rounded-lg border border-gray-300 bg-white/60 text-sm md:text-base text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-400 transition"
              value={form.confirm}
              onChange={handleChange}
              required
              autoComplete="new-password"
            />
            <button
              type="submit"
              className="w-full bg-gradient-to-r from-blue-500 to-purple-500 text-white py-2.5 md:py-3 rounded-lg font-semibold shadow hover:scale-105 transition disabled:opacity-60 text-sm md:text-base"
              disabled={loading}
            >
              {loading ? "Signing up..." : "Sign Up"}
            </button>
            <div className="text-center mt-3 md:mt-4">
              <button
                type="button"
                className="text-blue-600 hover:underline text-xs md:text-sm"
                onClick={() => setMode("login")}
                disabled={loading}
              >
                Already have an account? Login
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

export default AuthSplitScreen;
