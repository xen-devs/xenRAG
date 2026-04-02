import { type FormEvent, useState } from "react"
import { Navigate, useNavigate, useSearchParams } from "react-router-dom"
import { motion, AnimatePresence } from "framer-motion"
import { BrainCircuit, Loader2 } from "lucide-react"
import toast from "react-hot-toast"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { useAuth } from "@/contexts/auth-context"
import { PageTransition } from "@/components/page-transition"
import { signUp, signIn } from "@/lib/auth-api"

type AuthMode = "signin" | "signup"

function getErrorMessage(err: unknown, fallback: string): string {
  if (
    err &&
    typeof err === "object" &&
    "response" in err &&
    err.response &&
    typeof err.response === "object" &&
    "data" in err.response
  ) {
    const data = err.response.data as { detail?: string }
    return data.detail ?? fallback
  }
  return fallback
}

export function AuthPage() {
  const { isAuthenticated, user, login } = useAuth()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const [mode, setMode] = useState<AuthMode>(
    searchParams.get("tab") === "signup" ? "signup" : "signin",
  )
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [loading, setLoading] = useState(false)

  if (isAuthenticated && user) {
    if (user.orgs.length > 0) {
      return <Navigate to={`/org/${user.orgs[0].id}/chat`} replace />
    }
    return <Navigate to="/onboarding" replace />
  }

  function switchMode(newMode: AuthMode) {
    setMode(newMode)
    setName("")
    setEmail("")
    setPassword("")
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setLoading(true)

    try {
      if (mode === "signup") {
        const res = await signUp({ name, email, password })
        await login(res.access_token)
        toast.success("Account created successfully!")
        navigate("/onboarding")
      } else {
        const res = await signIn({ email, password })
        await login(res.access_token)
        toast.success("Welcome back!")
      }
    } catch (err: unknown) {
      toast.error(getErrorMessage(err, `${mode === "signin" ? "Sign in" : "Sign up"} failed. Please try again.`))
    } finally {
      setLoading(false)
    }
  }

  return (
    <PageTransition>
      <div className="flex min-h-dvh items-center justify-center">
        <Card className="w-full max-w-md shadow-none">
          <CardHeader className="space-y-2 pb-2">
            <div className="flex flex-col items-center gap-4">
              <div className="bg-primary/10 flex size-[72px] items-center justify-center rounded-2xl">
                <BrainCircuit className="text-primary size-9" />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col gap-6">
              <div className="flex flex-col items-center gap-2">
                <h1 className="text-xl font-bold">
                  {mode === "signup" ? "Create an Account" : "Welcome Back"}
                </h1>
                <p className="text-muted-foreground text-center text-sm">
                  {mode === "signup"
                    ? "Sign up to get started with xenRAG"
                    : "Login to your account to continue"}
                </p>
              </div>

              <AnimatePresence mode="wait">
                <motion.form
                  key={mode}
                  onSubmit={handleSubmit}
                  className="flex flex-col gap-4"
                  initial={{ opacity: 0, x: mode === "signup" ? 16 : -16 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: mode === "signup" ? -16 : 16 }}
                  transition={{ duration: 0.15 }}
                >
                  {mode === "signup" && (
                    <div className="flex flex-col gap-2">
                      <label htmlFor="name" className="text-sm font-medium">
                        Full Name
                      </label>
                      <Input
                        id="name"
                        type="text"
                        placeholder="Enter your full name"
                        required
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                      />
                    </div>
                  )}

                  <div className="flex flex-col gap-2">
                    <label htmlFor="email" className="text-sm font-medium">
                      Email
                    </label>
                    <Input
                      id="email"
                      type="email"
                      placeholder="Enter your email address"
                      required
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                    />
                  </div>

                  <div className="flex flex-col gap-2">
                    <label htmlFor="password" className="text-sm font-medium">
                      Password
                    </label>
                    <Input
                      id="password"
                      type="password"
                      placeholder={
                        mode === "signup"
                          ? "Min 6 characters"
                          : "Enter your password"
                      }
                      required
                      minLength={mode === "signup" ? 6 : undefined}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                    />
                  </div>

                  <div className="flex flex-col gap-3 pt-2">
                    <Button
                      type="submit"
                      className="w-full"
                      disabled={loading}
                    >
                      {loading && (
                        <Loader2 className="mr-2 size-4 animate-spin" />
                      )}
                      {mode === "signup" ? "Create Account" : "Sign In"}
                    </Button>
                    <p className="text-muted-foreground text-center text-sm">
                      {mode === "signup" ? (
                        <>
                          Already have an account?{" "}
                          <button
                            type="button"
                            onClick={() => switchMode("signin")}
                            className="text-primary cursor-pointer font-medium hover:underline"
                          >
                            Login
                          </button>
                        </>
                      ) : (
                        <>
                          Don&apos;t have an account?{" "}
                          <button
                            type="button"
                            onClick={() => switchMode("signup")}
                            className="text-primary cursor-pointer font-medium hover:underline"
                          >
                            Register
                          </button>
                        </>
                      )}
                    </p>
                  </div>
                </motion.form>
              </AnimatePresence>
            </div>
          </CardContent>
        </Card>
      </div>
    </PageTransition>
  )
}
