import { useState } from "react"
import { useNavigate } from "react-router-dom"
import { motion } from "framer-motion"
import { Building2, BrainCircuit, Check } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useAuth } from "@/contexts/auth-context"
import { PageTransition } from "@/components/page-transition"
import { cn } from "@/lib/utils"
import { CreateOrgForm } from "./create-org-form"

type OnboardingStep = "organization" | "complete"

const steps = [
  { id: "organization" as const, label: "Organization", icon: Building2 },
  { id: "complete" as const, label: "Get Started", icon: BrainCircuit },
]

export function OnboardingPage() {
  const navigate = useNavigate()
  const { user, login } = useAuth()
  const token = localStorage.getItem("auth_token")

  const hasOrg = user && user.orgs.length > 0
  const [currentStep, setCurrentStep] = useState<OnboardingStep>(
    hasOrg ? "complete" : "organization",
  )

  const effectiveStep = hasOrg && currentStep === "organization" ? "complete" : currentStep
  const currentStepIndex = steps.findIndex((s) => s.id === effectiveStep)

  function handleOrgSuccess() {
    // Re-fetch user data to get the new org
    if (token) {
      login(token).then(() => setCurrentStep("complete"))
    }
  }

  function handleGetStarted() {
    if (user && user.orgs.length > 0) {
      navigate(`/org/${user.orgs[0].id}/chat`)
    } else {
      navigate("/")
    }
  }

  return (
    <PageTransition>
      <div className="flex min-h-dvh flex-col items-center justify-center p-4">
        {/* Logo */}
        <div className="mb-8 flex items-center gap-2">
          <div className="bg-muted flex size-10 items-center justify-center rounded-lg">
            <BrainCircuit className="text-foreground size-5" />
          </div>
          <span className="text-2xl font-bold">xenRAG</span>
        </div>

        {/* Stepper */}
        <div className="mb-8 flex items-center gap-4">
          {steps.map((step, index) => {
            const isCompleted = index < currentStepIndex || (step.id === "organization" && hasOrg)
            const isCurrent = step.id === effectiveStep
            const StepIcon = step.icon

            return (
              <div key={step.id} className="flex items-center">
                <div className="flex flex-col items-center">
                  <motion.div
                    className={cn(
                      "flex h-10 w-10 items-center justify-center rounded-full border-2 transition-colors",
                      isCompleted && "border-primary bg-primary text-primary-foreground",
                      isCurrent && !isCompleted && "border-primary bg-primary/10 text-primary",
                      !isCompleted && !isCurrent && "border-muted-foreground/30 text-muted-foreground/50",
                    )}
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    {isCompleted ? <Check className="h-5 w-5" /> : <StepIcon className="h-5 w-5" />}
                  </motion.div>
                  <span
                    className={cn(
                      "mt-2 text-xs font-medium",
                      (isCurrent || isCompleted) && "text-primary",
                      !isCurrent && !isCompleted && "text-muted-foreground",
                    )}
                  >
                    {step.label}
                  </span>
                </div>
                {index < steps.length - 1 && (
                  <div
                    className={cn(
                      "mx-4 h-0.5 w-12 transition-colors",
                      isCompleted ? "bg-primary" : "bg-muted-foreground/30",
                    )}
                  />
                )}
              </div>
            )
          })}
        </div>

        {/* Step Content */}
        <motion.div
          key={effectiveStep}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25 }}
        >
          <Card className="w-full max-w-md">
            {effectiveStep === "organization" && (
              <>
                <CardHeader className="text-center">
                  <div className="bg-primary/10 mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full">
                    <Building2 className="text-primary h-6 w-6" />
                  </div>
                  <CardTitle>Create Your Organization</CardTitle>
                  <CardDescription>
                    Set up your organization to start using xenRAG. You can invite team members later.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <CreateOrgForm onSuccess={handleOrgSuccess} />
                </CardContent>
              </>
            )}

            {effectiveStep === "complete" && (
              <>
                <CardHeader className="text-center">
                  <div className="bg-primary/10 mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full">
                    <BrainCircuit className="text-primary h-6 w-6" />
                  </div>
                  <CardTitle>You're All Set!</CardTitle>
                  <CardDescription>
                    Your organization is ready. Start asking questions and get explainable, RAG-powered answers.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <button
                    type="button"
                    onClick={handleGetStarted}
                    className="bg-primary text-primary-foreground hover:bg-primary/90 inline-flex w-full items-center justify-center rounded-lg px-4 py-2 text-sm font-medium transition-colors"
                  >
                    Start Chatting
                  </button>
                </CardContent>
              </>
            )}
          </Card>
        </motion.div>
      </div>
    </PageTransition>
  )
}
