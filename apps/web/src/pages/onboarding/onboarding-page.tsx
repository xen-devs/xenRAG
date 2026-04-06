import { type FormEvent, useState } from "react"
import { useNavigate } from "react-router-dom"
import { motion } from "framer-motion"
import { Building2, BrainCircuit, Check, Loader2, Package } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { useAuth } from "@/contexts/auth-context"
import { PageTransition } from "@/components/page-transition"
import { cn } from "@/lib/utils"
import { createOrg } from "@/lib/auth-api"
import toast from "react-hot-toast"

type OnboardingStep = "organization" | "product"

const steps = [
  { id: "organization" as const, label: "Organization", icon: Building2 },
  { id: "product" as const, label: "Product Details", icon: Package },
]

export function OnboardingPage() {
  const navigate = useNavigate()
  const { user, login } = useAuth()
  const token = localStorage.getItem("auth_token")

  const hasOrg = user && user.orgs.length > 0
  const [currentStep, setCurrentStep] = useState<OnboardingStep>(
    hasOrg ? "product" : "organization",
  )
  const [orgName, setOrgName] = useState("")
  const [productName, setProductName] = useState("")
  const [description, setDescription] = useState("")
  const [loading, setLoading] = useState(false)

  const effectiveStep = hasOrg && currentStep === "organization" ? "product" : currentStep
  const currentStepIndex = steps.findIndex((s) => s.id === effectiveStep)

  function handleOrgContinue() {
    if (orgName.trim().length < 2) {
      toast.error("Organization name must be at least 2 characters")
      return
    }
    setCurrentStep("product")
  }

  async function handleSubmitProductDetails(e: FormEvent) {
    e.preventDefault()
    if (orgName.trim().length < 2) {
      toast.error("Organization name must be at least 2 characters")
      setCurrentStep("organization")
      return
    }

    setLoading(true)
    try {
      const createdOrg = await createOrg({
        name: orgName.trim(),
        product_name: productName.trim() || undefined,
        description: description.trim() || undefined,
      })
      if (token) {
        await login(token)
      }
      toast.success("Organization created successfully!")
      navigate(`/org/${createdOrg.id}/chat`)
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
        toast.error(data.detail ?? "Failed to create organization")
      } else {
        toast.error("Failed to create organization. Please try again.")
      }
    } finally {
      setLoading(false)
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
          className="w-full max-w-md"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25 }}
        >
          <Card className="w-full">
            {effectiveStep === "organization" && (
              <>
                <CardHeader className="text-center">
                  <div className="bg-primary/10 mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full">
                    <Building2 className="text-primary h-6 w-6" />
                  </div>
                  <CardTitle>Tell Us Your Organization</CardTitle>
                  <CardDescription>
                    Start with your organization name. We'll ask product details next.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-col gap-4">
                    <div className="flex flex-col gap-2">
                      <label htmlFor="org-name" className="text-sm font-medium">
                        Organization Name
                      </label>
                      <Input
                        id="org-name"
                        type="text"
                        placeholder="e.g., Xendev"
                        required
                        value={orgName}
                        onChange={(e) => setOrgName(e.target.value)}
                      />
                    </div>
                    <Button type="button" className="w-full" onClick={handleOrgContinue}>
                      Continue
                    </Button>
                  </div>
                </CardContent>
              </>
            )}

            {effectiveStep === "product" && (
              <>
                <CardHeader className="text-center">
                  <div className="bg-primary/10 mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full">
                    <Package className="text-primary h-6 w-6" />
                  </div>
                  <CardTitle>Product Details</CardTitle>
                  <CardDescription>
                    Add product context now, then get started.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <form onSubmit={handleSubmitProductDetails} className="flex flex-col gap-4">
                    <div className="flex flex-col gap-2">
                      <label htmlFor="product-name" className="text-sm font-medium">
                        Product Name
                      </label>
                      <Input
                        id="product-name"
                        type="text"
                        placeholder="e.g., Fire Stick"
                        autoComplete="off"
                        value={productName}
                        onChange={(e) => setProductName(e.target.value)}
                      />
                    </div>

                    <div className="flex flex-col gap-2">
                      <label htmlFor="description" className="text-sm font-medium">
                        Description
                      </label>
                      <textarea
                        id="description"
                        className="flex min-h-[80px] w-full rounded-lg border border-input bg-transparent px-2.5 py-2 text-sm text-foreground placeholder:text-muted-foreground outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30"
                        placeholder="Describe your product and feedback context..."
                        autoComplete="off"
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        maxLength={1000}
                      />
                    </div>

                    <Button type="submit" className="w-full" disabled={loading}>
                      {loading && <Loader2 className="mr-2 size-4 animate-spin" />}
                      Get Started
                    </Button>
                  </form>
                </CardContent>
              </>
            )}
          </Card>
        </motion.div>
      </div>
    </PageTransition>
  )
}
