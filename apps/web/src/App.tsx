import { Routes, Route, useLocation } from "react-router-dom"
import { AnimatePresence } from "framer-motion"
import { LandingPage } from "@/pages/landing/landing-page"
import { AuthPage } from "@/pages/auth/auth-page"
import { OnboardingPage } from "@/pages/onboarding/onboarding-page"
import { ChatPage } from "@/pages/chat/chat-page"
import { ProtectedRoute } from "@/components/protected-route"

export default function App() {
  const location = useLocation()

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<LandingPage />} />
        <Route path="/auth" element={<AuthPage />} />
        <Route
          path="/onboarding"
          element={
            <ProtectedRoute>
              <OnboardingPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/org/:orgId/chat"
          element={
            <ProtectedRoute>
              <ChatPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/org/:orgId/chat/:chatId"
          element={
            <ProtectedRoute>
              <ChatPage />
            </ProtectedRoute>
          }
        />
      </Routes>
    </AnimatePresence>
  )
}
