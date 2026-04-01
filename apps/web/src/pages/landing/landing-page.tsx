import { Link } from "react-router-dom"
import { motion } from "framer-motion"
import { ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { PageTransition } from "@/components/page-transition"

export function LandingPage() {
  return (
    <PageTransition>
      <div className="flex min-h-screen flex-col">
        {/* Navbar */}
        <header className="bg-background/80 sticky top-0 z-50 border-b backdrop-blur-sm">
          <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
            <Link to="/" className="flex items-center gap-2 font-semibold">
              <span>xenRAG</span>
            </Link>

            <div className="flex items-center gap-2">
              <Button variant="ghost" size="sm" asChild>
                <Link to="/auth?tab=signin">Sign In</Link>
              </Button>
              <Button size="sm" asChild>
                <Link to="/auth?tab=signup">Sign Up</Link>
              </Button>
            </div>
          </div>
        </header>

        {/* Hero */}
        <main className="flex flex-1 items-center justify-center px-4">
          <motion.div
            className="flex max-w-2xl flex-col items-center gap-6 text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.4 }}
          >
            <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
              AI Powered Customer Feedback Analysis
              <br />
            </h1>
            <div className="flex gap-3">
              <Button size="lg" asChild>
                <Link to="/auth?tab=signup">
                  Get Started
                  <ArrowRight className="ml-2 size-4" />
                </Link>
              </Button>
              <Button size="lg" variant="outline" asChild>
                <Link to="/auth?tab=signin">Sign In</Link>
              </Button>
            </div>
          </motion.div>
        </main>
      </div>
    </PageTransition>
  )
}
