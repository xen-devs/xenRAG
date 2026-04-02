import { type FormEvent, useState } from "react"
import { motion } from "framer-motion"
import { Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { signUp } from "@/lib/auth-api"
import { useAuth } from "@/contexts/auth-context"

export function SignUpForm() {
  const { login } = useAuth()
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [orgName, setOrgName] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)

    try {
      const res = await signUp({
        name,
        email,
        password,
        org_name: orgName,
      })
      await login(res.access_token)
    } catch (err: unknown) {
      if (
        err &&
        typeof err === "object" &&
        "response" in err &&
        err.response &&
        typeof err.response === "object" &&
        "data" in err.response
      ) {
        const data = err.response.data as { detail?: string }
        setError(data.detail ?? "Sign up failed")
      } else {
        setError("Sign up failed. Please try again.")
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <motion.form
      key="signup"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      transition={{ duration: 0.2 }}
      onSubmit={handleSubmit}
      className="flex flex-col gap-4"
    >
      <div className="flex flex-col gap-2">
        <label htmlFor="signup-name" className="text-sm font-medium">
          Name
        </label>
        <Input
          id="signup-name"
          type="text"
          placeholder="Your name"
          required
          value={name}
          onChange={(e) => {
            setName(e.target.value)
            setError(null)
          }}
        />
      </div>

      <div className="flex flex-col gap-2">
        <label htmlFor="signup-email" className="text-sm font-medium">
          Email
        </label>
        <Input
          id="signup-email"
          type="email"
          placeholder="you@example.com"
          required
          value={email}
          onChange={(e) => {
            setEmail(e.target.value)
            setError(null)
          }}
        />
      </div>

      <div className="flex flex-col gap-2">
        <label htmlFor="signup-password" className="text-sm font-medium">
          Password
        </label>
        <Input
          id="signup-password"
          type="password"
          placeholder="Min 6 characters"
          required
          minLength={6}
          value={password}
          onChange={(e) => {
            setPassword(e.target.value)
            setError(null)
          }}
        />
      </div>

      <div className="flex flex-col gap-2">
        <label htmlFor="signup-org" className="text-sm font-medium">
          Organization Name
        </label>
        <Input
          id="signup-org"
          type="text"
          placeholder="Your company or team"
          required
          value={orgName}
          onChange={(e) => {
            setOrgName(e.target.value)
            setError(null)
          }}
        />
      </div>

      {error && (
        <p className="text-destructive text-sm">{error}</p>
      )}

      <Button type="submit" className="w-full" disabled={loading}>
        {loading && <Loader2 className="mr-2 size-4 animate-spin" />}
        Create Account
      </Button>
    </motion.form>
  )
}
