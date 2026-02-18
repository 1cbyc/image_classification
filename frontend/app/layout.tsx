import type React from "react"
import type { Metadata } from "next"
import { Manrope, Poppins } from "next/font/google"
import "./globals.css"

const manrope = Manrope({
  subsets: ["latin"],
  variable: "--font-manrope",
})

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800", "900"],
  variable: "--font-poppins",
})

// Use Poppins for display/heading fonts as fallback
const calSans = poppins
const instrumentSans = poppins

export const metadata: Metadata = {
  title: "ReluRay - AI-Powered Medical Image Analysis",
  description: "Advanced AI-powered medical image analysis. Instant, accurate, and confidential chest X-ray analysis.",
  generator: 'reluray',
  icons: {
    icon: [
      { url: '/favicon.ico', sizes: 'any' },
      { url: '/favicon.svg', type: 'image/svg+xml' },
      { url: '/icon-light-32x32.png', media: '(prefers-color-scheme: light)', sizes: '32x32' },
      { url: '/icon-dark-32x32.png', media: '(prefers-color-scheme: dark)', sizes: '32x32' },
    ],
    apple: '/apple-icon.png',
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${manrope.variable} ${poppins.variable} font-sans antialiased`}>
        <div className="noise-overlay" aria-hidden="true" />
        {children}
        {/* Vercel Analytics removed for self-hosting */}
      </body>
    </html>
  )
}
