import { Card } from "@/app/components/ui/card";
import { Mail, Github, Twitter, MessageCircle } from "lucide-react";
import { useI18n } from "@/app/i18n/context";

export function ContactPage() {
  const { t } = useI18n();

  return (
    <div className="min-h-screen bg-background py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-12 text-center">
          <h1 className="text-4xl mb-4 text-primary">{t("contact.title")}</h1>
          <p className="text-xl text-muted-foreground">{t("contact.subtitle")}</p>
        </div>

        <div className="mb-12 grid grid-cols-1 gap-6 md:grid-cols-2">
          <Card className="bg-card p-6 shadow-lg transition-shadow hover:shadow-xl">
            <div className="flex items-start gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-background-info">
                <Mail className="h-6 w-6 text-text-info" />
              </div>
              <div>
                <h3 className="mb-2 text-primary">{t("contact.emailSupport")}</h3>
                <p className="mb-2 text-muted-foreground">{t("contact.emailSupportDesc")}</p>
                <a href="mailto:mittech.official@gmail.com" className="text-text-info hover:underline">
                  mittech.official@gmail.com
                </a>
              </div>
            </div>
          </Card>

          <Card className="bg-card p-6 shadow-lg transition-shadow hover:shadow-xl">
            <div className="flex items-start gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-background-info">
                <Github className="h-6 w-6 text-text-info" />
              </div>
              <div>
                <h3 className="mb-2 text-primary">GitHub</h3>
                <p className="mb-2 text-muted-foreground">{t("contact.githubDesc")}</p>
                <a
                  href="https://github.com/kaymid1/vietnamese_toxic_comment_detection_using_PhoBERT"
                  className="text-text-info hover:underline"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  github.com/vietnamese_toxic_comment_detection_using_PhoBERT
                </a>
              </div>
            </div>
          </Card>

          <Card className="bg-card p-6 shadow-lg transition-shadow hover:shadow-xl">
            <div className="flex items-start gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-background-info">
                <Twitter className="h-6 w-6 text-text-info" />
              </div>
              <div>
                <h3 className="mb-2 text-primary">Twitter/X</h3>
                <p className="mb-2 text-muted-foreground">{t("contact.twitterDesc")}</p>
                <a
                  href="https://twitter.com/viettoxic"
                  className="text-text-info hover:underline"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  @viettoxic
                </a>
              </div>
            </div>
          </Card>

          <Card className="bg-card p-6 shadow-lg transition-shadow hover:shadow-xl">
            <div className="flex items-start gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-background-info">
                <MessageCircle className="h-6 w-6 text-text-info" />
              </div>
              <div>
                <h3 className="mb-2 text-primary">Discord Community</h3>
                <p className="mb-2 text-muted-foreground">{t("contact.discordDesc")}</p>
                <a
                  href="https://discord.gg/viettoxic"
                  className="text-text-info hover:underline"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {t("contact.joinDiscord")}
                </a>
              </div>
            </div>
          </Card>
        </div>

        <Card className="mb-8 bg-card p-8 shadow-lg">
          <h2 className="mb-6 text-2xl text-primary">{t("contact.faqTitle")}</h2>

          <div className="space-y-6">
            <div>
              <h4 className="mb-2 text-primary">{t("contact.q1")}</h4>
              <p className="text-muted-foreground">{t("contact.a1")}</p>
            </div>

            <div>
              <h4 className="mb-2 text-primary">{t("contact.q2")}</h4>
              <p className="text-muted-foreground">{t("contact.a2")}</p>
            </div>

            <div>
              <h4 className="mb-2 text-primary">{t("contact.q3")}</h4>
              <p className="text-muted-foreground">{t("contact.a3")}</p>
            </div>

            <div>
              <h4 className="mb-2 text-primary">{t("contact.q4")}</h4>
              <p className="text-muted-foreground">{t("contact.a4")}</p>
            </div>

            <div>
              <h4 className="mb-2 text-primary">{t("contact.q5")}</h4>
              <p className="text-muted-foreground">{t("contact.a5")}</p>
            </div>
          </div>
        </Card>

        <Card className="bg-card p-8 shadow-lg">
          <h2 className="mb-4 text-2xl text-primary">{t("contact.researchTitle")}</h2>
          <p className="mb-4 text-muted-foreground">{t("contact.researchDesc")}</p>
          <div className="rounded-lg bg-background-secondary p-4">
            <h4 className="mb-2">{t("contact.collaborationAreas")}</h4>
            <ul className="space-y-2 text-sm text-foreground">
              <li className="flex items-start gap-2">
                <span className="mt-1 text-text-info">•</span>
                <span>{t("contact.c1")}</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-text-info">•</span>
                <span>{t("contact.c2")}</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-text-info">•</span>
                <span>{t("contact.c3")}</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-text-info">•</span>
                <span>{t("contact.c4")}</span>
              </li>
            </ul>
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            {t("contact.contactLine")} <a href="mailto:research@viettoxic.ai" className="text-text-info hover:underline">research@viettoxic.ai</a>
          </p>
        </Card>
      </div>
    </div>
  );
}
