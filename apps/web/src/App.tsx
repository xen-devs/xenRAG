import { Routes, Route, Navigate, useLocation } from "react-router-dom"
import { AnimatePresence } from "framer-motion"
import { LandingPage } from "@/pages/landing/landing-page"
import { AuthPage } from "@/pages/auth/auth-page"
import { OnboardingPage } from "@/pages/onboarding/onboarding-page"
import { ChatPage } from "@/pages/chat/chat-page"
import { ProtectedRoute } from "@/components/protected-route"
import { SettingsLayout } from "@/pages/settings/settings-layout"
import { MyAccountPage } from "@/pages/settings/my-account-page"
import { KnowledgeBasePage } from "@/pages/settings/knowledge-base-page"

function getRouteKey(pathname: string) {
  const chatMatch = pathname.match(/^\/org\/[^/]+\/chat/)
  if (chatMatch) return chatMatch[0]
  return pathname
}

export default function App() {
  const location = useLocation()

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={getRouteKey(location.pathname)}>
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
        <Route
          path="/org/:orgId/settings"
          element={
            <ProtectedRoute>
              <SettingsLayout />
            </ProtectedRoute>
          }
        >
          <Route index element={<Navigate to="my-account" replace />} />
          <Route path="my-account" element={<MyAccountPage />} />
          <Route path="knowledge-base" element={<KnowledgeBasePage />} />
        </Route>
      </Routes>
    </AnimatePresence>
  )
}
